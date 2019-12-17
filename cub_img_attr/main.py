from torchvision import datasets, transforms

import time
import random
import torch
import os
import numpy as np
from model import EncoderA, DecoderA, EncoderB, DecoderB
from datasets import datasets
import torch.nn as nn
import sys

sys.path.append('../')
import probtorch
import util
import visdom

# ------------------------------------------------
# training parameters

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=2, metavar='N',
                        help='run_id')
    parser.add_argument('--run_desc', type=str, default='',
                        help='run_id desc')
    parser.add_argument('--n_privateA', type=int, default=636,
                        help='size of the latent embedding of privateA')
    parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--ckpt_epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default: 1e-3]')

    parser.add_argument('--beta', type=str, default='1,10,1',
                        help='beta for TC. [img, attr, label]')
    parser.add_argument('--lamb', type=str, default='1,500,5000',
                        help='lambda for reconst. [img, attr, label')

    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed for get_paired_data')
    parser.add_argument('--wseed', type=int, default=0, metavar='N',
                        help='random seed for weight')

    parser.add_argument('--ckpt_path', type=str, default='../weights/cub',
                        help='save and load path for ckpt')
    parser.add_argument('--gpu', type=str, default='',
                        help='cuda')
    parser.add_argument('--outgpu', type=int, default=-1,
                        help='outgpu')
    parser.add_argument('--data_path', type=str, default='../../data/cub/CUB_200_2011/CUB_200_2011/',
                        help='data path')
    # visdom
    parser.add_argument('--viz_on',
                        default=False, type=probtorch.util.str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_port',
                        default=8002, type=int, help='visdom port number')
    args = parser.parse_args()

# ------------------------------------------------


EPS = 1e-9

if len(args.gpu) > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    GPU = [int(elt) for elt in args.gpu.split(',')]
    print('GPU:', GPU)
else:
    GPU = []
CUDA = torch.cuda.is_available()

beta = [float(i) for i in args.beta.split(',')]
lamb = [float(i) for i in args.lamb.split(',')]

# path parameters
MODEL_NAME = 'cub_ia-run_id%d-privA%02ddim-lamb%s-beta%s-lr%s-bs%s-wseed%s-seed%s' % (
    args.run_id, args.n_privateA, '_'.join([str(elt) for elt in lamb]), '_'.join([str(elt) for elt in beta]),
    args.lr, args.batch_size, args.wseed, args.seed)

if len(args.run_desc) > 1:
    desc_file = os.path.join(args.ckpt_path, 'run_id' + str(args.run_id) + '.txt')
    with open(desc_file, 'w') as outfile:
        outfile.write(args.run_desc)

N_PIXELS = 3 * 128 * 128
TEMP = 0.66
NUM_SAMPLES = 1
import pickle

path = args.data_path
ATTR_PRIOR = pickle.load(open(path + "attributes/attr_prior.pkl", "rb"))
for i in range(len(ATTR_PRIOR)):
    ATTR_PRIOR[i] = torch.FloatTensor(ATTR_PRIOR[i])
    if CUDA:
        ATTR_PRIOR[i] = ATTR_PRIOR[i].cuda()

primary_attr = ['eye_color', 'bill_length', 'shape', 'breast_pattern', 'belly_pattern', 'bill_shape',
                'bill_color', 'throat_color', 'crown_color', 'forehead_color', 'underparts_color', 'primary_color',
                'breast_color', 'wing_color']
# original
primary_attr = ['eye_color', 'bill_length', 'size', 'shape', 'breast_pattern', 'belly_pattern', 'bill_shape',
                'bill_color', 'throat_color', 'crown_color', 'forehead_color', 'underparts_color', 'primary_color',
                'breast_color', 'wing_color']
# regressor + stat 10
primary_attr = ['eye_color', 'bill_length', 'leg_color', 'bill_shape',
                'bill_color', 'throat_color', 'crown_color', 'forehead_color',
                'belly_color', 'wing_color']

# regressor + stat 13
primary_attr = ['eye_color', 'bill_length', 'shape', 'bill_shape', 'head_pattern',
                'bill_color', 'throat_color', 'crown_color', 'forehead_color', 'breast_color',
                'belly_color', 'wing_color', 'leg_color']

ATTR_IDX = []
ATTR_DIM = []
N_ATTR = len(primary_attr)
TRAIN_CLASSES = np.genfromtxt(path + 'attributes/trainvalids.txt', delimiter='\n', dtype=int)

attributes = np.genfromtxt(path + 'attributes/attr.txt', delimiter='\n', dtype=str)
total_attr = []
for i in range(attributes.shape[0]):
    attr = attributes[i].split("::")[0]
    total_attr.append(attr)
    if attr in primary_attr:
        ATTR_IDX.append(i)
        ATTR_DIM.append(len(attributes[i].split("::")[1].split(',')) + 1)

ATTR_PRIOR = [ATTR_PRIOR[i] for i in ATTR_IDX]
print(primary_attr)
print(ATTR_IDX)
print(np.array(total_attr)[ATTR_IDX])
print(len(ATTR_IDX))


# visdom setup
def viz_init():
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llA'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llB'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llA_test'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llB_test'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['total_losses'])


def visualize_line():
    data = LINE_GATHER.data
    recon_A = torch.Tensor(data['recon_A'])
    recon_B = torch.Tensor(data['recon_B'])

    recon_A_test = torch.Tensor(data['recon_A_test'])
    recon_B_test = torch.Tensor(data['recon_B_test'])

    epoch = torch.Tensor(data['epoch'])
    test_total_loss = torch.Tensor(data['test_total_loss'])
    total_loss = torch.Tensor(data['total_loss'])

    total_losses = torch.tensor(np.stack([total_loss, test_total_loss], -1))

    VIZ.line(
        X=epoch, Y=recon_A, env=MODEL_NAME + '/lines',
        win=WIN_ID['llA'], update='append',
        opts=dict(xlabel='epoch', ylabel='loglike',
                  title='LL of modalA', legend=['A', 'poeA', 'crA'])
    )
    VIZ.line(
        X=epoch, Y=recon_B, env=MODEL_NAME + '/lines',
        win=WIN_ID['llB'], update='append',
        opts=dict(xlabel='epoch', ylabel='loglike',
                  title='LL of modalB', legend=['B', 'poeB', 'crB'])
    )

    VIZ.line(
        X=epoch, Y=recon_B_test, env=MODEL_NAME + '/lines',
        win=WIN_ID['llB_test'], update='append',
        opts=dict(xlabel='epoch', ylabel='loglike',
                  title='LL of modalB test', legend=['B', 'crB'])
    )

    VIZ.line(
        X=epoch, Y=recon_A_test, env=MODEL_NAME + '/lines',
        win=WIN_ID['llA_test'], update='append',
        opts=dict(xlabel='epoch', ylabel='loglike',
                  title='LL of modalA test', legend=['A', 'crA'])
    )

    VIZ.line(
        X=epoch, Y=total_losses, env=MODEL_NAME + '/lines',
        win=WIN_ID['total_losses'], update='append',
        opts=dict(xlabel='epoch', ylabel='loss',
                  title='Total Loss', legend=['train_loss', 'test_loss'])
    )


if args.viz_on:
    WIN_ID = dict(
        llA='win_llA', llB='win_llB',
        total_losses='win_total_losses',
        llB_test='win_llB_test', llA_test='win_llA_test'
    )
    LINE_GATHER = probtorch.util.DataGather(
        'epoch', 'recon_A', 'recon_B',
        'total_loss', 'test_total_loss', 'recon_A_test', 'recon_B_test'
    )
    VIZ = visdom.Visdom(port=args.viz_port)
    viz_init()

train_data = torch.utils.data.DataLoader(datasets(path, ATTR_IDX, train=True, crop=1.2), batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=len(GPU))
test_data = torch.utils.data.DataLoader(datasets(path, ATTR_IDX, train=False, crop=1.2), batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=len(GPU))

BIAS_TRAIN = (train_data.dataset.__len__() - 1) / (args.batch_size - 1)
BIAS_TEST = (test_data.dataset.__len__() - 1) / (args.batch_size - 1)


def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())


encA = EncoderA(args.wseed, zPrivate_dim=args.n_privateA, zSharedAttr_dim=ATTR_DIM)
decA = DecoderA(args.wseed, zPrivate_dim=args.n_privateA, zSharedAttr_dim=ATTR_DIM)
encB = EncoderB(args.wseed, zSharedAttr_dim=ATTR_DIM)
decB = DecoderB(args.wseed, zSharedAttr_dim=ATTR_DIM)

if CUDA:
    encA.cuda()
    decA.cuda()
    encB.cuda()
    decB.cuda()
    cuda_tensors(encA)
    cuda_tensors(decA)
    cuda_tensors(encB)
    cuda_tensors(decB)
    if len(args.gpu) > 2:
        print('multi: ' + args.gpu)
        encA = nn.DataParallel(encA)
        decA = nn.DataParallel(decA)
        encB = nn.DataParallel(encB)
        decB = nn.DataParallel(decB)

optimizer = torch.optim.Adam(
    list(encB.parameters()) + list(decB.parameters()) + list(
        encA.parameters()) + list(decA.parameters()),
    lr=args.lr)


def elbo(q, pA, pB, lamb, beta, bias=1.0, train=True):
    # attribute from each of image, attr modality
    reconst_loss_poeA = reconst_loss_crA = reconst_loss_poeB = reconst_loss_crB = None
    reconst_loss_A = None

    sharedA_attr = []
    sharedB_attr = []
    poe_attr = []
    for i in range(N_ATTR):
        sharedA_attr.append('sharedA_attr' + str(i))
        sharedB_attr.append('sharedB_attr' + str(i))
        poe_attr.append('poe_attr' + str(i))

    reconst_loss_A, kl_A = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_own'],
                                                               latents=np.concatenate(
                                                                   [['privateA'], sharedA_attr]),
                                                               sample_dim=0,
                                                               batch_dim=1,
                                                               beta=(1, beta[0], 1), bias=bias)

    reconst_loss_B, kl_B = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['attr_own'],
                                                               latents=sharedB_attr,
                                                               sample_dim=0, batch_dim=1,
                                                               beta=(1, beta[1], 1), bias=bias)

    reconst_loss_crA, kl_crA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_cross'],
                                                                   latents=np.concatenate(
                                                                       [['privateA'], sharedB_attr]),
                                                                   sample_dim=0,
                                                                   batch_dim=1,
                                                                   beta=(1, beta[0], 1), bias=bias)

    reconst_loss_crB, kl_crB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['attr_cross'],
                                                                   latents=sharedA_attr,
                                                                   sample_dim=0, batch_dim=1,
                                                                   beta=(1, beta[1], 1), bias=bias)

    if train:
        reconst_loss_poeA, kl_poeA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_poe'],
                                                                         latents=np.concatenate(
                                                                             [['privateA'], poe_attr]),
                                                                         sample_dim=0,
                                                                         batch_dim=1,
                                                                         beta=(1, beta[0], 1), bias=bias)

        reconst_loss_poeB, kl_poeB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['attr_poe'],
                                                                         latents=poe_attr,
                                                                         sample_dim=0, batch_dim=1,
                                                                         beta=(1, beta[1], 1), bias=bias)

        loss = (lamb[0] * reconst_loss_A - kl_A) + (lamb[1] * reconst_loss_B - kl_B) + \
               (lamb[0] * reconst_loss_poeA - kl_poeA) + (lamb[1] * reconst_loss_poeB - kl_poeB) + \
               (lamb[0] * reconst_loss_crA - kl_crA) + (lamb[1] * reconst_loss_crB - kl_crB)


    else:
        loss = 3 * ((lamb[0] * reconst_loss_A - kl_A) + (lamb[1] * reconst_loss_B - kl_B))

    return -loss, [reconst_loss_A, reconst_loss_poeA, reconst_loss_crA], [reconst_loss_B, reconst_loss_poeB,
                                                                          reconst_loss_crB]


def train(data, encA, decA, encB, decB, optimizer):
    epoch_elbo = 0.0
    epoch_recA = epoch_rec_poeA = epoch_rec_crA = 0.0
    epoch_recB = epoch_rec_poeB = epoch_rec_crB = 0.0
    encA.train()
    encB.train()
    decA.train()
    decB.train()

    N = 0
    for b, (images, attr, _) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += 1
            attributes = []
            for i in range(args.batch_size):
                concat_all_attr = []
                for j in range(N_ATTR):
                    concat_all_attr.append(attr[j][i])
                attributes.append(torch.cat(concat_all_attr, dim=0))
            attributes = torch.stack(attributes).float()
            optimizer.zero_grad()
            if CUDA:
                images = images.cuda()
                attributes = attributes.cuda()
                for i in range(len(attr)):
                    attr[i] = attr[i].cuda()
            # encode
            q = encA(images, num_samples=NUM_SAMPLES)
            q = encB(attributes, num_samples=NUM_SAMPLES, q=q)

            ## poe of attributes from modal A & B ##
            for i in range(N_ATTR):
                prior_logit = torch.log(ATTR_PRIOR[i] + EPS)
                poe_logit = q['sharedA_attr' + str(i)].dist.logits + q[
                    'sharedB_attr' + str(i)].dist.logits + prior_logit
                q.concrete(logits=poe_logit,
                           temperature=TEMP,
                           name='poe_attr' + str(i))

            sharedA_attr = []
            sharedB_attr = []
            poe_attr = []

            for i in range(N_ATTR):
                poe_attr.append('poe_attr' + str(i))
                sharedA_attr.append('sharedA_attr' + str(i))
                sharedB_attr.append('sharedB_attr' + str(i))

            # decode attr
            shared_dist = {'poe': poe_attr, 'cross': sharedA_attr, 'own': sharedB_attr}

            pB = decB(attr, shared_dist, ATTR_PRIOR, q=q, num_samples=NUM_SAMPLES)

            # decode img
            shared_dist = {'poe': poe_attr, 'cross': sharedB_attr, 'own': sharedA_attr}
            pA = decA(images, shared_dist, ATTR_PRIOR, q=q,
                      num_samples=NUM_SAMPLES)

            # loss
            loss, recA, recB = elbo(q, pA, pB, lamb=lamb, beta=beta, bias=BIAS_TRAIN)
            loss.backward()
            optimizer.step()
            if CUDA:
                loss = loss.cpu()
                recA[0] = recA[0].cpu()
                recB[0] = recB[0].cpu()

            epoch_elbo += loss.item()
            epoch_recA += recA[0].item()
            epoch_rec_poeA += recA[1].item()
            epoch_rec_crA += recA[2].item()

            epoch_recB += recB[0].item()
            epoch_rec_poeB += recB[1].item()
            epoch_rec_crB += recB[2].item()

    return epoch_elbo / N, [epoch_recA / N, epoch_rec_poeA / N, epoch_rec_crA / N], \
           [epoch_recB / N, epoch_rec_poeB / N, epoch_rec_crB / N]


def test(data, encA, decA, encB, decB, epoch):
    encA.eval()
    decA.eval()
    encB.eval()
    decB.eval()
    epoch_elbo = 0.0
    epoch_correct = epoch_correct_from_img = epoch_correct_from_attr = 0
    N = 0
    for b, (images, attr, _) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += 1
            attributes = []
            for i in range(args.batch_size):
                concat_all_attr = []
                for j in range(N_ATTR):
                    concat_all_attr.append(attr[j][i])
                attributes.append(torch.cat(concat_all_attr, dim=0))
            attributes = torch.stack(attributes).float()

            optimizer.zero_grad()
            if CUDA:
                images = images.cuda()
                attributes = attributes.cuda()
                for i in range(len(attr)):
                    attr[i] = attr[i].cuda()
            # encode
            q = encA(images, num_samples=NUM_SAMPLES)
            q = encB(attributes, num_samples=NUM_SAMPLES, q=q)

            sharedA_attr = []
            sharedB_attr = []
            for i in range(N_ATTR):
                sharedA_attr.append('sharedA_attr' + str(i))
                sharedB_attr.append('sharedB_attr' + str(i))

            # decode attr
            shared_dist = {'cross': sharedA_attr, 'own': sharedB_attr}
            pB = decB(attr, shared_dist, ATTR_PRIOR, q=q,
                      num_samples=NUM_SAMPLES)

            # decode img
            shared_dist = {'cross': sharedB_attr, 'own': sharedA_attr}
            pA = decA(images, shared_dist, ATTR_PRIOR, q=q,
                      num_samples=NUM_SAMPLES)

            # loss
            batch_elbo, recA, recB = elbo(q, pA, pB, lamb=lamb, beta=beta, bias=BIAS_TEST, train=False)
            ######

            if CUDA:
                batch_elbo = batch_elbo.cpu()
            epoch_elbo += batch_elbo.item()
    return epoch_elbo / N, recA, recB


####
def save_ckpt(e):
    torch.save(encA, '%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(encB, '%s/%s-encB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(decA, '%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(decB, '%s/%s-decB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))


####
if args.ckpt_epochs > 0:
    if CUDA:
        encA = torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs))
        encB = torch.load('%s/%s-encB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs))
        decA = torch.load('%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs))
        decB = torch.load('%s/%s-decB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs))
    else:
        encA = torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs), map_location='cpu')
        encB = torch.load('%s/%s-encB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs), map_location='cpu')
        decA = torch.load('%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs), map_location='cpu')
        decB = torch.load('%s/%s-decB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs), map_location='cpu')

for e in range(args.ckpt_epochs, args.epochs):
    train_start = time.time()
    train_elbo, rec_lossA, rec_lossB = train(train_data, encA, decA, encB, decB, optimizer)
    train_end = time.time()
    test_start = time.time()
    test_elbo, recon_A_test, recon_B_test = test(test_data, encA, decA, encB, decB, e)

    if args.viz_on:
        LINE_GATHER.insert(epoch=e,
                           test_total_loss=test_elbo,
                           total_loss=train_elbo,
                           recon_A=rec_lossA,
                           recon_B=rec_lossB,
                           recon_A_test=[recon_A_test[0], recon_A_test[2]],
                           recon_B_test=[recon_B_test[0], recon_B_test[2]]
                           )
        visualize_line()
        LINE_GATHER.flush()

    test_end = time.time()
    if (e + 1) % 10 == 0 or e + 1 == args.epochs:
        save_ckpt(e + 1)
        util.evaluation.save_traverse_cub_ia(e, test_data, encA, decA, CUDA, MODEL_NAME, ATTR_DIM,
                                             fixed_idxs=[277, 342, 658, 1570, 2233, 2388, 2880, 1344, 2750, 1111,
                                                         300],
                                             private=False)  # 2880
        util.evaluation.save_traverse_cub_ia(e, train_data, encA, decA, CUDA, MODEL_NAME, ATTR_DIM,
                                             fixed_idxs=[215, 336, 502, 537, 575, 4288, 1000, 2400, 1220, 3002, 3312],
                                             private=False)
    print('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e, cross_attr %0.3f (%ds)' % (
        e, train_elbo, train_end - train_start,
        test_elbo, recon_B_test[2], test_end - test_start))

if args.ckpt_epochs == args.epochs:
    # util.evaluation.save_recon_cub(args.epochs, train_data, encA, decA, encB, CUDA, MODEL_NAME, ATTR_DIM,
    #                                   fixed_idxs=[130, 215, 502, 537, 4288, 1000, 2400, 1220, 3002, 3312])
    util.evaluation.save_traverse_cub_ia(args.epochs, test_data, encA, decA, CUDA, MODEL_NAME, ATTR_DIM,
                                         fixed_idxs=[277, 342, 658, 1570, 2233, 2388, 2880, 1344, 2750, 1111,
                                                     300],
                                         private=False)  # 2880
    # train
    util.evaluation.save_traverse_cub_ia(args.epochs, train_data, encA, decA, CUDA, MODEL_NAME, ATTR_DIM,
                                         fixed_idxs=[215, 336, 502, 537, 575, 4288, 1000, 2400, 1220, 3002, 3312],
                                         private=False)


else:
    save_ckpt(args.epochs)
