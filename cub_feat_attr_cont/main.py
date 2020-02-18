from torchvision import datasets, transforms

import time
import random
import torch
import os
import numpy as np
from model import EncoderA, DecoderA, EncoderB, DecoderB, DecoderA2, LINEAR_LOGSOFTMAX
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
    parser.add_argument('--run_id', type=int, default=3, metavar='N',
                        help='run_id')
    parser.add_argument('--run_desc', type=str, default='',
                        help='run_id desc')
    parser.add_argument('--n_privateA', type=int, default=64,
                        help='size of the latent embedding of privateA')
    parser.add_argument('--n_shared', type=int, default=28,
                        help='size of the latent embedding of shared')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--ckpt_epochs', type=int, default=120, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--epochs', type=int, default=120, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default: 1e-3]')

    parser.add_argument('--beta', type=str, default='1,1',
                        help='beta for TC. [img, attr, label]')
    parser.add_argument('--lamb', type=str, default='1,10',
                        help='lambda for reconst. [img, attr, label')

    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed for get_paired_data')
    parser.add_argument('--wseed', type=int, default=0, metavar='N',
                        help='random seed for weight')

    parser.add_argument('--ckpt_path', type=str, default='../weights/cub_feat_attr_cont',
                        help='save and load path for ckpt')
    parser.add_argument('--gpu', type=str, default='',
                        help='cuda')
    parser.add_argument('--outgpu', type=int, default=-1,
                        help='outgpu')
    parser.add_argument('--num_hidden', type=int, default=0,
                        help='num_hidden')

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
MODEL_NAME = 'cub_feat_attr_cont-run_id%d-privA%02ddim-shared%02ddim-lamb%s-beta%s-lr%s-bs%s-wseed%s-seed%s-num_hidden%s' % (
    args.run_id, args.n_privateA, args.n_shared, '_'.join([str(elt) for elt in lamb]),
    '_'.join([str(elt) for elt in beta]),
    args.lr, args.batch_size, args.wseed, args.seed, args.num_hidden)

if len(args.run_desc) > 1:
    print(args.run_desc)
    desc_file = os.path.join(args.ckpt_path, 'run_id' + str(args.run_id) + '.txt')
    with open(desc_file, 'w') as outfile:
        outfile.write(args.run_desc)

N_PIXELS = 3 * 128 * 128
TEMP = 0.66
NUM_SAMPLES = 1
import pickle

path = args.data_path

ATTR_IDX = []
ATTR_DIM = []
N_ATTR = 312
TRAIN_CLASSES = np.genfromtxt(path + 'cvpr2016/trainvalids.txt', delimiter='\n', dtype=int)

attributes = np.genfromtxt(path + 'feature/attr.txt', delimiter='\n', dtype=str)
total_attr = []
for i in range(attributes.shape[0]):
    ATTR_IDX.append(i)
    ATTR_DIM.append(len(attributes[i].split("::")[1].split(',')))

print(ATTR_IDX)
print(len(ATTR_IDX))
print(sum(ATTR_DIM))


# visdom setup
def viz_init():
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llA'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llB'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llA_test'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llB_test'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llA_val'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llB_val'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['total_losses'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['acc'])


def visualize_line():
    data = LINE_GATHER.data
    recon_A = torch.Tensor(data['recon_A'])
    recon_B = torch.Tensor(data['recon_B'])

    recon_A_test = torch.Tensor(data['recon_A_test'])
    recon_B_test = torch.Tensor(data['recon_B_test'])

    recon_A_val = torch.Tensor(data['recon_A_val'])
    recon_B_val = torch.Tensor(data['recon_B_val'])

    epoch = torch.Tensor(data['epoch'])
    test_total_loss = torch.Tensor(data['test_total_loss'])
    val_total_loss = torch.Tensor(data['val_total_loss'])
    total_loss = torch.Tensor(data['total_loss'])

    te_acc = torch.Tensor(data['te_acc'])
    tr_acc = torch.Tensor(data['tr_acc'])
    val_acc = torch.Tensor(data['val_acc'])

    # te_f1 = torch.Tensor(data['te_f1'])
    # tr_f1 = torch.Tensor(data['tr_f1'])
    # val_f1 = torch.Tensor(data['val_f1'])


    total_losses = torch.tensor(np.stack([total_loss, test_total_loss, val_total_loss], -1))
    acc = torch.tensor(np.stack([tr_acc, val_acc, te_acc], -1))
    # f1 = torch.tensor(np.stack([tr_f1, val_f1, te_f1], -1))

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
        X=epoch, Y=recon_B_val, env=MODEL_NAME + '/lines',
        win=WIN_ID['llB_val'], update='append',
        opts=dict(xlabel='epoch', ylabel='loglike',
                  title='LL of modalB val', legend=['B', 'crB'])
    )

    VIZ.line(
        X=epoch, Y=recon_A_val, env=MODEL_NAME + '/lines',
        win=WIN_ID['llA_val'], update='append',
        opts=dict(xlabel='epoch', ylabel='loglike',
                  title='LL of modalA val', legend=['A', 'crA'])
    )

    VIZ.line(
        X=epoch, Y=total_losses, env=MODEL_NAME + '/lines',
        win=WIN_ID['total_losses'], update='append',
        opts=dict(xlabel='epoch', ylabel='loss',
                  title='Total Loss', legend=['train_loss', 'test_loss', 'val_loss'])
    )

    VIZ.line(
        X=epoch, Y=acc, env=MODEL_NAME + '/lines',
        win=WIN_ID['acc'], update='append',
        opts=dict(xlabel='epoch', ylabel='loss',
                  title='Acc', legend=['tr_acc', 'val_acc', 'te_acc'])
    )

    # VIZ.line(
    #     X=epoch, Y=f1, env=MODEL_NAME + '/lines',
    #     win=WIN_ID['f1'], update='append',
    #     opts=dict(xlabel='epoch', ylabel='loss',
    #               title='F1-score', legend=['tr_f1', 'val_f1', 'te_f1'])
    # )


if args.viz_on:
    WIN_ID = dict(
        llA='win_llA', llB='win_llB',
        total_losses='win_total_losses',
        llB_test='win_llB_test', llA_test='win_llA_test',
        llB_val='win_llB_val', llA_val='win_llA_val', acc='win_acc'
    )
    LINE_GATHER = probtorch.util.DataGather(
        'epoch', 'recon_A', 'recon_B',
        'total_loss', 'test_total_loss', 'recon_A_test', 'recon_B_test',
        'val_total_loss', 'recon_A_val', 'recon_B_val', 'tr_acc', 'te_acc', 'val_acc'
    )
    VIZ = visdom.Visdom(port=args.viz_port)
    viz_init()

train_data = torch.utils.data.DataLoader(datasets(path, ATTR_IDX, train=True, crop=1.2), batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=len(GPU))
test_data = torch.utils.data.DataLoader(datasets(path, ATTR_IDX, train=False, crop=1.2), batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=len(GPU))

val_data = torch.utils.data.DataLoader(datasets(path, ATTR_IDX, train=True, crop=1.2, val=True),
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=len(GPU))

BIAS_TRAIN = (train_data.dataset.__len__() - 1) / (args.batch_size - 1)
BIAS_TEST = (test_data.dataset.__len__() - 1) / (args.batch_size - 1)


def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())


encA = EncoderA(args.wseed, zPrivate_dim=args.n_privateA, zShared_dim=args.n_shared, num_hidden=args.num_hidden)
decA = DecoderA(args.wseed, zPrivate_dim=args.n_privateA, zShared_dim=args.n_shared, num_hidden=args.num_hidden)
encB = EncoderB(args.wseed, zShared_dim=args.n_shared, num_hidden=args.num_hidden)
decB = DecoderB(args.wseed, zShared_dim=args.n_shared, num_hidden=args.num_hidden)

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


def elbo(q, pA, pB=None, lamb=[1., 1.], beta=[1., 1.], bias=1.0, train=True):
    # attribute from each of image, attr modality
    reconst_loss_poeA = reconst_loss_crA = reconst_loss_poeB = reconst_loss_crB = None
    reconst_loss_A = None

    reconst_loss_A, kl_A = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_own'],
                                                               latents=['privateA', 'sharedA'],
                                                               sample_dim=0,
                                                               batch_dim=1,
                                                               beta=(1, beta[0], 1), bias=bias)

    reconst_loss_B, kl_B = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['attr_own'],
                                                               latents=['sharedB'],
                                                               sample_dim=0, batch_dim=1,
                                                               beta=(1, beta[1], 1), bias=bias)

    reconst_loss_crA, kl_crA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_cross'],
                                                                   latents=['privateA', 'sharedB'],
                                                                   sample_dim=0,
                                                                   batch_dim=1,
                                                                   beta=(1, beta[0], 1), bias=bias)

    reconst_loss_crB, kl_crB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['attr_cross'],
                                                                   latents=['sharedA'],
                                                                   sample_dim=0, batch_dim=1,
                                                                   beta=(1, beta[1], 1), bias=bias)

    if train:
        reconst_loss_poeA, kl_poeA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_poe'],
                                                                         latents=['privateA', 'poe'],
                                                                         sample_dim=0,
                                                                         batch_dim=1,
                                                                         beta=(1, beta[0], 1), bias=bias)

        reconst_loss_poeB, kl_poeB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['attr_poe'],
                                                                         latents=['poe'],
                                                                         sample_dim=0, batch_dim=1,
                                                                         beta=(1, beta[1], 1), bias=bias)

        loss = (lamb[0] * reconst_loss_A - kl_A) + (lamb[1] * reconst_loss_B - kl_B) + \
               (lamb[0] * reconst_loss_poeA - kl_poeA) + (lamb[1] * reconst_loss_poeB - kl_poeB) + \
               (lamb[0] * reconst_loss_crA - kl_crA) + (lamb[1] * reconst_loss_crB - kl_crB)


    else:
        if pB:
            loss = 3 * ((lamb[0] * reconst_loss_A - kl_A) + (lamb[1] * reconst_loss_B - kl_B))
        else:
            reconst_loss_B = None
            loss = 3 * ((lamb[0] * reconst_loss_A - kl_A))

    return -loss, [reconst_loss_A, reconst_loss_poeA, reconst_loss_crA], [reconst_loss_B, reconst_loss_poeB,
                                                                          reconst_loss_crB]


def train(data, encA, decA, encB, decB, optimizer):
    epoch_elbo = epoch_distance = 0.0
    epoch_recA = epoch_rec_poeA = epoch_rec_crA = 0.0
    epoch_recB = epoch_rec_poeB = epoch_rec_crB = 0.0
    encA.train()
    encB.train()
    decA.train()
    decB.train()

    N = 0
    for b, (images, attributes, _) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += 1
            attributes = attributes.float()
            optimizer.zero_grad()
            if CUDA:
                images = images.cuda()
                attributes = attributes.cuda()
            # encode
            q = encA(images, num_samples=NUM_SAMPLES)
            q = encB(attributes, num_samples=NUM_SAMPLES, q=q)

            ## poe ##
            mu_poe, std_poe = probtorch.util.apply_poe(CUDA, q['sharedA'].dist.loc, q['sharedA'].dist.scale,
                                                       q['sharedB'].dist.loc, q['sharedB'].dist.scale)
            q.normal(mu_poe,
                     std_poe,
                     name='poe')

            # decode attr
            shared_dist = {'poe': 'poe', 'cross': 'sharedA', 'own': 'sharedB'}
            pB = decB(attributes, shared_dist, q=q, num_samples=NUM_SAMPLES)

            # decode img
            shared_dist = {'poe': 'poe', 'cross': 'sharedB', 'own': 'sharedA'}
            pA = decA(images, shared_dist, q=q, num_samples=NUM_SAMPLES)

            # loss
            loss, recA, recB = elbo(q, pA, pB, lamb=lamb, beta=beta, bias=BIAS_TRAIN)
            loss.backward()
            optimizer.step()
            if CUDA:
                loss = loss.cpu()
                for i in range(3):
                    recA[i] = recA[i].cpu()
                    recB[i] = recB[i].cpu()

            epoch_elbo += loss.item()
            epoch_recA += recA[0].item()
            epoch_rec_poeA += recA[1].item()
            epoch_rec_crA += recA[2].item()

            epoch_recB += recB[0].item()
            epoch_rec_poeB += recB[1].item()
            epoch_rec_crB += recB[2].item()

            distance = torch.sqrt(torch.sum((q['sharedA'].dist.loc - q['sharedB'].dist.loc) ** 2, dim=1) + \
                                  torch.sum((q['sharedA'].dist.scale - q['sharedB'].dist.scale) ** 2, dim=1))

            distance = distance.sum()
            if CUDA:
                distance = distance.cup()

            epoch_distance += distance.item()

    return epoch_elbo / N, [epoch_recA / N, epoch_rec_poeA / N, epoch_rec_crA / N], \
           [epoch_recB / N, epoch_rec_poeB / N, epoch_rec_crB / N], epoch_distance / N


def test(data, encA, decA, encB, decB, epoch):
    encA.eval()
    decA.eval()
    encB.eval()
    decB.eval()
    epoch_elbo = epoch_distance = 0.0
    epoch_recA = epoch_rec_crA = 0.0
    epoch_recB = epoch_rec_crB = 0.0
    N = 0
    for b, (images, attributes, _) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += 1
            attributes = attributes.float()
            optimizer.zero_grad()
            if CUDA:
                images = images.cuda()
                attributes = attributes.cuda()
            # encode
            q = encA(images, num_samples=NUM_SAMPLES)
            q = encB(attributes, num_samples=NUM_SAMPLES, q=q)

            # decode attr
            shared_dist = {'cross': 'sharedA', 'own': 'sharedB'}
            pB = decB(attributes, shared_dist, q=q,
                      num_samples=NUM_SAMPLES)

            # decode img
            shared_dist = {'cross': 'sharedB', 'own': 'sharedA'}
            pA = decA(images, shared_dist, q=q,
                      num_samples=NUM_SAMPLES)

            # loss
            loss, recA, recB = elbo(q, pA, pB, lamb=lamb, beta=beta, bias=BIAS_TEST, train=False)
            ######

            if CUDA:
                loss = loss.cpu()
                for i in [0, 2]:
                    recA[i] = recA[i].cpu()
                    recB[i] = recB[i].cpu()

            epoch_elbo += loss.item()
            epoch_recA += recA[0].item()
            epoch_rec_crA += recA[2].item()

            epoch_recB += recB[0].item()
            epoch_rec_crB += recB[2].item()
            epoch_elbo += loss.item()

            distance = torch.sqrt(torch.sum((q['sharedA'].dist.loc - q['sharedB'].dist.loc) ** 2, dim=1) + \
                                  torch.sum((q['sharedA'].dist.scale - q['sharedB'].dist.scale) ** 2, dim=1))

            distance = distance.sum()
            if CUDA:
                distance = distance.cup()

            epoch_distance += distance

    return epoch_elbo / N, [epoch_recA / N, epoch_rec_crA / N], \
           [epoch_recB / N, epoch_rec_crB / N], epoch_distance / N


####
def save_ckpt(e):
    torch.save(encA, '%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(encB, '%s/%s-encB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(decA, '%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(decB, '%s/%s-decB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))

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
    train_elbo, rec_lossA, rec_lossB, tr_dist = train(train_data, encA, decA, encB, decB, optimizer)
    train_end = time.time()

    val_start = time.time()
    val_elbo, recon_A_val, recon_B_val, val_dist = test(val_data, encA, decA, encB, decB, e)
    val_end = time.time()

    test_start = time.time()
    test_elbo, recon_A_test, recon_B_test, te_dist = test(test_data, encA, decA, encB, decB, e)
    test_end = time.time()

    if args.viz_on:
        LINE_GATHER.insert(epoch=e,
                           test_total_loss=test_elbo,
                           val_total_loss=val_elbo,
                           total_loss=train_elbo,
                           recon_A=rec_lossA,
                           recon_B=rec_lossB,
                           recon_A_test=recon_A_test,
                           recon_B_test=recon_B_test,
                           recon_A_val=recon_A_val,
                           recon_B_val=recon_B_val,
                           tr_acc=tr_dist,
                           val_acc=val_dist,
                           te_acc=te_dist
                           )
        visualize_line()
        LINE_GATHER.flush()

    if (e + 1) % 10 == 0 or e + 1 == args.epochs:
        save_ckpt(e + 1)
        # util.evaluation.save_traverse_cub_ia2(e, test_data, encA, decA, CUDA, MODEL_NAME, ATTR_DIM,
        #                                       fixed_idxs=[277, 342, 658, 1570, 2233, 2388, 2880, 1344, 2750, 1111],
        #                                       private=False)  # 2880
        # util.evaluation.save_traverse_cub_ia2(e, train_data, encA, decA, CUDA, MODEL_NAME, ATTR_DIM,
        #                                       fixed_idxs=[336, 502, 537, 575, 4288, 1000, 2400, 1220, 3002, 3312],
        #                                       private=False)
    print('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e, cross_attr %0.3f (%ds)' % (
        e, train_elbo, train_end - train_start,
        test_elbo, recon_B_test[1], test_end - test_start))


###############################

def classifier(data, encA, decA, encB, decB):
    encA.eval()
    decA.eval()
    encB.eval()
    decB.eval()
    epoch_elbo = epoch_distance = 0.0
    epoch_recA = epoch_rec_crA = 0.0
    epoch_recB = epoch_rec_crB = 0.0
    N = 0
    clf = LINEAR_LOGSOFTMAX(64, 50)
    for b, (images, attributes, _) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += 1
            attributes = attributes.float()
            optimizer.zero_grad()
            if CUDA:
                images = images.cuda()
                attributes = attributes.cuda()
            # encode
            q = encA(images, num_samples=NUM_SAMPLES)
            q = encB(attributes, num_samples=NUM_SAMPLES, q=q)

            # decode attr
            shared_dist = {'cross': 'sharedA', 'own': 'sharedB'}
            pB = decB(attributes, shared_dist, q=q,
                      num_samples=NUM_SAMPLES)

            # decode img
            shared_dist = {'cross': 'sharedB', 'own': 'sharedA'}
            pA = decA(images, shared_dist, q=q,
                      num_samples=NUM_SAMPLES)

            # loss
            loss, recA, recB = elbo(q, pA, pB, lamb=lamb, beta=beta, bias=BIAS_TEST, train=False)
            ######

            if CUDA:
                loss = loss.cpu()
                for i in [0, 2]:
                    recA[i] = recA[i].cpu()
                    recB[i] = recB[i].cpu()

            epoch_elbo += loss.item()
            epoch_recA += recA[0].item()
            epoch_rec_crA += recA[2].item()

            epoch_recB += recB[0].item()
            epoch_rec_crB += recB[2].item()
            epoch_elbo += loss.item()

            img_mean = q['sharedA'].dist.loc
            attr_mean = q['sharedB'].dist.loc
            img_std = q['sharedA'].dist.scale
            attr_std = q['sharedB'].dist.scale

            if CUDA:
                img_mean = img_mean.cpu()
                attr_mean = attr_mean.cpu()
                img_std = img_std.cpu()
                attr_std = attr_std.cpu()

            distance = torch.sqrt(torch.sum((img_mean - attr_mean) ** 2, dim=1) + \
                                  torch.sum((img_std - attr_std) ** 2, dim=1))

            distance = distance.sum()
            epoch_distance += distance

    return epoch_elbo / N, [epoch_recA / N, epoch_rec_crA / N], \
           [epoch_recB / N, epoch_rec_crB / N], epoch_distance / N


###############################

if args.ckpt_epochs == args.epochs:

    ae_decA = DecoderA2(0)
    if CUDA:
        ae_decA = torch.load('../weights/cub_img_ae/cub-img-ae-run_id3-bs64-decA_epoch200.rar')
    else:
        ae_decA = torch.load('../weights/cub_img_ae/cub-img-ae-run_id3-bs64-decA_epoch200.rar', map_location='cpu')

    fixed_idxs = [658, 1570, 2233, 2456, 2880, 1344, 2750, 1800, 1111, 300, 700,
                  1270, 2133, 2856, 2680, 1300]
    imgs = [0] * len(fixed_idxs)
    attributes = [0] * len(fixed_idxs)

    for i, idx in enumerate(fixed_idxs):
        imgs[i], attributes[i] = test_data.dataset.__getitem__(idx)[:2]
        if CUDA:
            imgs[i] = imgs[i].cuda()
            imgs[i] = imgs[i].squeeze(0)
            attributes[i] = attributes[i].cuda()
    imgs = torch.stack(imgs, dim=0)
    attributes = torch.stack(attributes, dim=0)

    q = encA(imgs, num_samples=1)
    q = encB(attributes, num_samples=NUM_SAMPLES, q=q)
    latents = [q['privateA'].dist.loc, q['sharedA'].dist.loc]
    recon_img_feat = decA.forward2(latents)
    recon_img = ae_decA.forward(recon_img_feat)

    from torchvision.utils import save_image


    def mkdirs(path):
        if not os.path.exists(path):
            os.makedirs(path)


    out_dir = os.path.join('../output/' + MODEL_NAME, str(args.ckpt_epochs), str(fixed_idxs))
    mkdirs(out_dir)
    # save_image(imgs,
    #            str(os.path.join(out_dir, 'gt_image.png')), nrow=int(np.sqrt(imgs.shape[0])))
    save_image(recon_img,
               str(os.path.join(out_dir, 'recon_image.png')), nrow=int(np.sqrt(recon_img.shape[0])))

else:
    save_ckpt(args.epochs)
