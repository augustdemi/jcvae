from torchvision import datasets, transforms

import time
import random
import torch
import os
import numpy as np
from model import EncoderA, DecoderA, EncoderB, DecoderB, EncoderC, DecoderC, DecoderA2
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
MODEL_NAME = 'cub-run_id%d-privA%02ddim-lamb%s-beta%s-lr%s-bs%s-wseed%s-seed%s' % (
    args.run_id, args.n_privateA, '_'.join([str(elt) for elt in lamb]), '_'.join([str(elt) for elt in beta]),
    args.lr, args.batch_size, args.wseed, args.seed)

if len(args.run_desc) > 1:
    desc_file = os.path.join(args.ckpt_path, 'run_id' + str(args.run_id) + '.txt')
    with open(desc_file, 'w') as outfile:
        outfile.write(args.run_desc)

N_CLASSES = 200
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
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['test_acc'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['total_losses'])


def visualize_line():
    data = LINE_GATHER.data
    recon_A = torch.Tensor(data['recon_A'])
    recon_B = torch.Tensor(data['recon_B'])
    recon_C = torch.Tensor(data['recon_C'])

    recon_B_test = torch.Tensor(data['rec_lossB_testset'])
    recon_C_test = torch.Tensor(data['rec_lossC_testset'])

    epoch = torch.Tensor(data['epoch'])
    train_acc = torch.Tensor(data['train_acc'])
    test_acc = torch.Tensor(data['test_acc'])
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
        X=epoch, Y=recon_C, env=MODEL_NAME + '/lines',
        win=WIN_ID['llC'], update='append',
        opts=dict(xlabel='epoch', ylabel='loglike',
                  title='LL of modalC', legend=['C', 'poeC', 'crC'])
    )

    VIZ.line(
        X=epoch, Y=recon_B_test, env=MODEL_NAME + '/lines',
        win=WIN_ID['llB_test'], update='append',
        opts=dict(xlabel='epoch', ylabel='loglike',
                  title='LL of modalB test in training', legend=['B', 'crB'])
    )

    VIZ.line(
        X=epoch, Y=recon_C_test, env=MODEL_NAME + '/lines',
        win=WIN_ID['llC_test'], update='append',
        opts=dict(xlabel='epoch', ylabel='loglike',
                  title='LL of modalC test in training', legend=['C', 'poeC', 'crC'])
    )

    VIZ.line(
        X=epoch, Y=train_acc, env=MODEL_NAME + '/lines',
        win=WIN_ID['train_acc'], update='append',
        opts=dict(xlabel='epoch', ylabel='accuracy',
                  title='Train Accuracy', legend=['acc', 'img_acc', 'attr_acc'])
    )

    VIZ.line(
        X=epoch, Y=test_acc, env=MODEL_NAME + '/lines',
        win=WIN_ID['test_acc'], update='append',
        opts=dict(xlabel='epoch', ylabel='accuracy',
                  title='Test Accuracy', legend=['acc', 'img_acc', 'attr_acc'])
    )

    VIZ.line(
        X=epoch, Y=total_losses, env=MODEL_NAME + '/lines',
        win=WIN_ID['total_losses'], update='append',
        opts=dict(xlabel='epoch', ylabel='loss',
                  title='Total Loss', legend=['train_loss', 'test_loss'])
    )


if args.viz_on:
    WIN_ID = dict(
        llA='win_llA', llB='win_llB', llC='win_llC', train_acc='win_train_acc', test_acc='win_test_acc',
        total_losses='win_total_losses',
        llB_test='win_llB_test', llC_test='win_llC_test'
    )
    LINE_GATHER = probtorch.util.DataGather(
        'epoch', 'recon_A', 'recon_B', 'recon_C',
        'total_loss', 'test_total_loss', 'train_acc', 'test_acc', 'rec_lossB_testset', 'rec_lossC_testset'
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


encA = EncoderA(args.wseed, zPrivate_dim=args.n_privateA, zSharedAttr_dim=ATTR_DIM, zSharedLabel_dim=N_CLASSES)
decA = DecoderA(args.wseed, zPrivate_dim=args.n_privateA, zSharedAttr_dim=ATTR_DIM, zSharedLabel_dim=N_CLASSES)
encB = EncoderB(args.wseed, zSharedAttr_dim=ATTR_DIM, zSharedLabel_dim=N_CLASSES)
decB = DecoderB(args.wseed, zSharedAttr_dim=ATTR_DIM, zSharedLabel_dim=N_CLASSES)
encC = EncoderC(args.wseed, zSharedLabel_dim=N_CLASSES)
decC = DecoderC(args.wseed, zSharedLabel_dim=N_CLASSES)

if CUDA:
    encA.cuda()
    decA.cuda()
    encB.cuda()
    decB.cuda()
    encC.cuda()
    decC.cuda()
    cuda_tensors(encA)
    cuda_tensors(decA)
    cuda_tensors(encB)
    cuda_tensors(decB)
    cuda_tensors(encC)
    cuda_tensors(decC)
    if len(args.gpu) > 2:
        print('multi: ' + args.gpu)
        encA = nn.DataParallel(encA)
        decA = nn.DataParallel(decA)
        encB = nn.DataParallel(encB)
        decB = nn.DataParallel(decB)
        encC = nn.DataParallel(encC)
        decC = nn.DataParallel(decC)
        # device = torch.device("cuda:0")
        # encA.to(device)
        # encA = nn.DataParallel(encA, device_ids=GPU, output_device=args.outgpu)
        # decA = nn.DataParallel(decA, device_ids=GPU, output_device=args.outgpu)
        # encB = nn.DataParallel(encB, device_ids=GPU, output_device=args.outgpu)
        # decB = nn.DataParallel(decB, device_ids=GPU, output_device=args.outgpu)

optimizer = torch.optim.Adam(
    list(encC.parameters()) + list(decC.parameters()) + list(encB.parameters()) + list(decB.parameters()) + list(
        encA.parameters()) + list(decA.parameters()),
    lr=args.lr)


def elbo(q, pA, pB, pC, lamb, beta, bias=1.0, train=True):
    # attribute from each of image, attr modality
    reconst_loss_poeA = reconst_loss_crA = reconst_loss_poeB = reconst_loss_crB = None
    reconst_loss_A = reconst_loss_crC = reconst_loss_poeC = None

    sharedA_attr = []
    sharedB_attr = []
    poe_attr = []
    for i in range(N_ATTR):
        sharedA_attr.append('sharedA_attr' + str(i))
        sharedB_attr.append('sharedB_attr' + str(i))
        poe_attr.append('poe_attr' + str(i))

    reconst_loss_B, kl_B = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['attr_own'],
                                                               latents=np.concatenate(
                                                                   [sharedB_attr, ['sharedB_label']]),
                                                               sample_dim=0, batch_dim=1,
                                                               beta=(1, beta[1], 1), bias=bias)
    reconst_loss_C, kl_C = probtorch.objectives.mws_tcvae.elbo(q, pC, pC['label_own'],
                                                               latents=['sharedC_label'],
                                                               sample_dim=0, batch_dim=1,
                                                               beta=(1, beta[2], 1), bias=bias)

    if train:
        # both train-train and train-test
        reconst_loss_poeC, kl_poeC = probtorch.objectives.mws_tcvae.elbo(q, pC, pC['label_poe'],
                                                                         latents=['poe_label'],
                                                                         sample_dim=0, batch_dim=1,
                                                                         beta=(1, beta[2], 1), bias=bias)

        reconst_loss_crC_fromB, kl_crC_fromB = probtorch.objectives.mws_tcvae.elbo(q, pC, pC['label_crossB'],
                                                                                   latents=['sharedB_label'],
                                                                                   sample_dim=0, batch_dim=1,
                                                                                   beta=(1, beta[2], 1), bias=bias)

        # train-train
        if pA is not None:
            reconst_loss_A, kl_A = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_own'],
                                                                       latents=np.concatenate(
                                                                           [['privateA'], sharedA_attr,
                                                                            ['sharedA_label']]),
                                                                       sample_dim=0,
                                                                       batch_dim=1,
                                                                       beta=(1, beta[0], 1), bias=bias)
            reconst_loss_poeA, kl_poeA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_poe'],
                                                                             latents=np.concatenate(
                                                                                 [['privateA'], poe_attr,
                                                                                  ['poe_label']]),
                                                                             sample_dim=0,
                                                                             batch_dim=1,
                                                                             beta=(1, beta[0], 1), bias=bias)
            reconst_loss_crA, kl_crA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_cross'],
                                                                           latents=np.concatenate(
                                                                               [['privateA'], sharedB_attr,
                                                                                ['sharedC_label']]),
                                                                           sample_dim=0,
                                                                           batch_dim=1,
                                                                           beta=(1, beta[0], 1), bias=bias)

            reconst_loss_poeB, kl_poeB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['attr_poe'],
                                                                             latents=np.concatenate(
                                                                                 [poe_attr, ['poe_label']]),
                                                                             sample_dim=0, batch_dim=1,
                                                                             beta=(1, beta[1], 1), bias=bias)

            reconst_loss_crB, kl_crB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['attr_cross'],
                                                                           latents=np.concatenate(
                                                                               [sharedA_attr, ['sharedC_label']]),
                                                                           sample_dim=0, batch_dim=1,
                                                                           beta=(1, beta[1], 1), bias=bias)

            reconst_loss_crC_fromA, kl_crC_fromA = probtorch.objectives.mws_tcvae.elbo(q, pC, pC['label_crossA'],
                                                                                       latents=['sharedA_label'],
                                                                                       sample_dim=0, batch_dim=1,
                                                                                       beta=(1, beta[2], 1), bias=bias)

            # loss = (lamb[0] * reconst_loss_A / N_PIXELS - kl_A) + (lamb[1] * reconst_loss_B / N_ATTR - kl_B) + (
            # lamb[2] * reconst_loss_C - kl_C) + \
            #        (lamb[0] * reconst_loss_poeA / N_PIXELS - kl_poeA) + (
            #        lamb[1] * reconst_loss_poeB / N_ATTR - kl_poeB) + (
            #        lamb[2] * reconst_loss_poeC - kl_poeC) + \
            #        (lamb[0] * reconst_loss_crA / N_PIXELS - kl_crA) + (lamb[1] * reconst_loss_crB / N_ATTR - kl_crB) + \
            #        0.5 * ((lamb[2] * reconst_loss_crC_fromB - kl_crC_fromB) + (
            #        lamb[2] * reconst_loss_crC_fromA - kl_crC_fromA))
            loss = (lamb[0] * reconst_loss_A - kl_A) + (lamb[1] * reconst_loss_B - kl_B) + (
            lamb[2] * reconst_loss_C - kl_C) + \
                   (lamb[0] * reconst_loss_poeA - kl_poeA) + (lamb[1] * reconst_loss_poeB - kl_poeB) + (
                   lamb[2] * reconst_loss_poeC - kl_poeC) + \
                   (lamb[0] * reconst_loss_crA - kl_crA) + (lamb[1] * reconst_loss_crB - kl_crB) + \
                   0.5 * ((lamb[2] * reconst_loss_crC_fromB - kl_crC_fromB) + (
                   lamb[2] * reconst_loss_crC_fromA - kl_crC_fromA))

            reconst_loss_crC = 0.5 * (reconst_loss_crC_fromB + reconst_loss_crC_fromA)
        else:
            reconst_loss_crB, kl_crB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['attr_cross'],
                                                                           latents=np.concatenate(
                                                                               [sharedB_attr, ['sharedC_label']]),
                                                                           sample_dim=0, batch_dim=1,
                                                                           beta=(1, beta[1], 1), bias=bias)
            # loss = 1.5 * (
            # 1.5 * ((lamb[1] * reconst_loss_B / N_ATTR - kl_B) + (lamb[1] * reconst_loss_crB / N_ATTR - kl_crB)) + \
            # ((lamb[2] * reconst_loss_C - kl_C) + \
            #                (lamb[2] * reconst_loss_poeC - kl_poeC) + \
            #                (lamb[2] * reconst_loss_crC_fromB - kl_crC_fromB)))
            loss = 1.5 * (1.5 * ((lamb[1] * reconst_loss_B - kl_B) + (lamb[1] * reconst_loss_crB - kl_crB)) + \
                          ((lamb[2] * reconst_loss_C - kl_C) + \
                           (lamb[2] * reconst_loss_poeC - kl_poeC) + \
                           (lamb[2] * reconst_loss_crC_fromB - kl_crC_fromB)))
            reconst_loss_crC = reconst_loss_crC_fromB

    else:
        reconst_loss_A, kl_A = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_own'],
                                                                   latents=np.concatenate(
                                                                       [['privateA'], sharedA_attr]),
                                                                   sample_dim=0,
                                                                   batch_dim=1,
                                                                   beta=(1, beta[0], 1), bias=bias)

        # loss = 3 * (
        #     (lamb[0] * reconst_loss_A / N_PIXELS - kl_A) + (lamb[1] * reconst_loss_B / N_ATTR - kl_B) + (
        #     lamb[2] * reconst_loss_C - kl_C))
        loss = 3 * (
            (lamb[0] * reconst_loss_A - kl_A) + (lamb[1] * reconst_loss_B - kl_B) + (lamb[2] * reconst_loss_C - kl_C))

    return -loss, [reconst_loss_A, reconst_loss_poeA, reconst_loss_crA], [reconst_loss_B, reconst_loss_poeB,
                                                                          reconst_loss_crB], [reconst_loss_C,
                                                                                              reconst_loss_poeC,
                                                                                              reconst_loss_crC]


def train(data, encA, decA, encB, decB, encC, decC, optimizer):
    epoch_elbo = 0.0
    epoch_recA = epoch_rec_poeA = epoch_rec_crA = 0.0
    epoch_recB = epoch_rec_poeB = epoch_rec_crB = 0.0
    epoch_recC = epoch_rec_poeC = epoch_rec_crC = 0.0
    encA.train()
    encB.train()
    encC.train()
    decA.train()
    decB.train()
    decC.train()

    N = 0
    epoch_correct = epoch_correct_from_img = epoch_correct_from_attr = 0

    torch.autograd.set_detect_anomaly(True)
    for b, (images, attr, labels) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += 1
            labels_onehot = torch.zeros(args.batch_size, N_CLASSES)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS)
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
                labels_onehot = labels_onehot.cuda()
                attributes = attributes.cuda()
                for i in range(len(attr)):
                    attr[i] = attr[i].cuda()
            # encode
            q = encA(images, num_samples=NUM_SAMPLES)
            q = encB(attributes, num_samples=NUM_SAMPLES, q=q)
            q = encC(labels_onehot, num_samples=NUM_SAMPLES, q=q)

            ## poe of label from modal A, B & C ##
            prior_logit = torch.zeros_like(
                q['sharedC_label'].dist.logits)  # prior for label is the concrete dist. of uniform dist.
            # if CUDA:
            #     prior_logit = prior_logit.cuda()
            poe_logit = q['sharedA_label'].dist.logits + q['sharedB_label'].dist.logits + q[
                'sharedC_label'].dist.logits + prior_logit
            q.concrete(logits=poe_logit,
                       temperature=TEMP,
                       name='poe_label')

            ## poe of attributes from modal A & B ##
            for i in range(N_ATTR):
                prior_logit = torch.log(ATTR_PRIOR[i] + EPS)
                poe_logit = q['sharedA_attr' + str(i)].dist.logits + q[
                    'sharedB_attr' + str(i)].dist.logits + prior_logit
                q.concrete(logits=poe_logit,
                           temperature=TEMP,
                           name='poe_attr' + str(i))

            # decode label
            shared_dist = {'poe': 'poe_label', 'crossA': 'sharedA_label', 'crossB': 'sharedB_label',
                           'own': 'sharedC_label'}
            pC = decC(labels_onehot, shared_dist, q=q,
                      num_samples=NUM_SAMPLES)

            # decode attr
            shared_dist = {'poe': [[], 'poe_label'], 'cross': [[], 'sharedC_label'],
                           'own': [[], 'sharedB_label']}
            for i in range(N_ATTR):
                shared_dist['poe'][0].append('poe_attr' + str(i))
                shared_dist['cross'][0].append('sharedA_attr' + str(i))
                shared_dist['own'][0].append('sharedB_attr' + str(i))
            pB = decB(attr, shared_dist, ATTR_PRIOR, q=q,
                      num_samples=NUM_SAMPLES)

            # decode img
            shared_dist = {'poe': [[], 'poe_label'], 'cross': [[], 'sharedC_label'], 'own': [[], 'sharedA_label']}
            for i in range(N_ATTR):
                shared_dist['poe'][0].append('poe_attr' + str(i))
                shared_dist['cross'][0].append('sharedB_attr' + str(i))
                shared_dist['own'][0].append('sharedA_attr' + str(i))
            pA = decA(images, shared_dist, ATTR_PRIOR, q=q,
                      num_samples=NUM_SAMPLES)

            # loss
            loss, recA, recB, recC = elbo(q, pA, pB, pC, lamb=lamb, beta=beta, bias=BIAS_TRAIN)
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

            epoch_recC += recC[0].item()
            epoch_rec_poeC += recC[1].item()
            epoch_rec_crC += recC[2].item()

            # accuracy
            epoch_correct += pC['acc_label_own'].loss.sum().item() / args.batch_size
            epoch_correct_from_img += pC['acc_label_crossA'].loss.sum().item() / args.batch_size
            epoch_correct_from_attr += pC['acc_label_crossB'].loss.sum().item() / args.batch_size
    acc = [1 + epoch_correct / N, 1 + epoch_correct_from_img / N, 1 + epoch_correct_from_attr / N]


    return epoch_elbo / N, [epoch_recA / N, epoch_rec_poeA / N, epoch_rec_crA / N], \
           [epoch_recB / N, epoch_rec_poeB / N, epoch_rec_crB / N], \
           [epoch_recC / N, epoch_rec_poeC / N, epoch_rec_crC / N], acc


def train_testset(data, encB, decB, encC, decC, optimizer):
    epoch_elbo = 0.0
    epoch_recB = epoch_rec_crB = 0.0
    epoch_recC = epoch_rec_poeC = epoch_rec_crC = 0.0
    encB.train()
    encC.train()
    decB.train()
    decC.train()

    N = 0
    torch.autograd.set_detect_anomaly(True)
    # for test set, image modality is not trained
    for b, (_, attr, labels) in enumerate(data):
        if labels.size()[0] == args.batch_size:
            N += 1
            labels_onehot = torch.zeros(args.batch_size, N_CLASSES)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS)
            attributes = []
            for i in range(args.batch_size):
                concat_all_attr = []
                for j in range(N_ATTR):
                    concat_all_attr.append(attr[j][i])
                attributes.append(torch.cat(concat_all_attr, dim=0))
            attributes = torch.stack(attributes).float()
            optimizer.zero_grad()
            if CUDA:
                labels_onehot = labels_onehot.cuda()
                attributes = attributes.cuda()
                for i in range(len(attr)):
                    attr[i] = attr[i].cuda()
            # encode
            q = encB(attributes, num_samples=NUM_SAMPLES)
            q = encC(labels_onehot, num_samples=NUM_SAMPLES, q=q)

            ## poe of label from modal A, B & C ##
            prior_logit = torch.zeros_like(
                q['sharedC_label'].dist.logits)  # prior for label is the concrete dist. of uniform dist.
            poe_logit = q['sharedB_label'].dist.logits + q['sharedC_label'].dist.logits + prior_logit
            q.concrete(logits=poe_logit,
                       temperature=TEMP,
                       name='poe_label')

            # decode label
            shared_dist = {'partial_poe': 'poe_label', 'crossB': 'sharedB_label', 'own': 'sharedC_label'}
            pC = decC(labels_onehot, shared_dist, q=q,
                      num_samples=NUM_SAMPLES, acc=False)

            # decode attr
            shared_dist = {'cross': [[], 'sharedC_label'],
                           'own': [[], 'sharedB_label']}
            for i in range(N_ATTR):
                shared_dist['cross'][0].append('sharedB_attr' + str(i))
                shared_dist['own'][0].append('sharedB_attr' + str(i))
            pB = decB(attr, shared_dist, ATTR_PRIOR, q=q,
                      num_samples=NUM_SAMPLES)

            # loss
            loss, _, recB, recC = elbo(q, None, pB, pC, lamb=lamb, beta=beta, bias=BIAS_TEST)
            loss.backward()
            optimizer.step()
            if CUDA:
                loss = loss.cpu()
                for i in range(3):
                    recC[i] = recC[i].cpu()
                recB[0] = recB[0].cpu()
                recB[2] = recB[2].cpu()

            epoch_elbo += loss.item()
            epoch_recB += recB[0].item()
            epoch_rec_crB += recB[2].item()

            epoch_recC += recC[0].item()
            epoch_rec_poeC += recC[1].item()
            epoch_rec_crC += recC[2].item()

    return epoch_elbo / N, [epoch_recB / N, epoch_rec_crB / N], [epoch_recC / N, epoch_rec_poeC / N, epoch_rec_crC / N]


def test(data, encA, decA, encB, decB, epoch):
    encA.eval()
    decA.eval()
    encB.eval()
    decB.eval()
    encC.eval()
    decC.eval()
    epoch_elbo = 0.0
    epoch_correct = epoch_correct_from_img = epoch_correct_from_attr = 0
    N = 0
    for b, (images, attr, labels) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += 1
            labels_onehot = torch.zeros(args.batch_size, N_CLASSES)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS)
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
                labels_onehot = labels_onehot.cuda()
                attributes = attributes.cuda()
                for i in range(len(attr)):
                    attr[i] = attr[i].cuda()
            # encode
            q = encA(images, num_samples=NUM_SAMPLES)
            q = encB(attributes, num_samples=NUM_SAMPLES, q=q)
            q = encC(labels_onehot, num_samples=NUM_SAMPLES, q=q)

            # decode label
            shared_dist = {'crossA': 'sharedA_label', 'crossB': 'sharedB_label', 'own': 'sharedC_label'}

            pC = decC(labels_onehot, shared_dist, q=q,
                      num_samples=NUM_SAMPLES)

            # decode attr
            shared_dist = {'own': [[], 'sharedB_label']}
            for i in range(N_ATTR):
                shared_dist['own'][0].append('sharedB_attr' + str(i))
            pB = decB(attr, shared_dist, ATTR_PRIOR, q=q,
                      num_samples=NUM_SAMPLES)

            # decode img
            shared_dist = {'own': [[], 'sharedA_label']}
            for i in range(N_ATTR):
                shared_dist['own'][0].append('sharedA_attr' + str(i))
            pA = decA(images, shared_dist, ATTR_PRIOR, q=q,
                      num_samples=NUM_SAMPLES)

            # loss
            batch_elbo, _, _, _ = elbo(q, pA, pB, pC, lamb=lamb, beta=beta, bias=BIAS_TEST, train=False)
            ######

            if CUDA:
                batch_elbo = batch_elbo.cpu()
            epoch_elbo += batch_elbo.item()
            epoch_correct += pC['acc_label_own'].loss.sum().item() / args.batch_size
            epoch_correct_from_img += pC['acc_label_crossA'].loss.sum().item() / args.batch_size
            epoch_correct_from_attr += pC['acc_label_crossB'].loss.sum().item() / args.batch_size
    acc = [1 + epoch_correct / N, 1 + epoch_correct_from_img / N, 1 + epoch_correct_from_attr / N]
    return epoch_elbo / N, acc


####
def save_ckpt(e):
    torch.save(encA, '%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(encB, '%s/%s-encB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(encC, '%s/%s-encC_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(decA, '%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(decB, '%s/%s-decB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(decC, '%s/%s-decC_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))


####
if args.ckpt_epochs > 0:
    if CUDA:
        encA = torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs))
        encB = torch.load('%s/%s-encB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs))
        encC = torch.load('%s/%s-encC_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs))
        decA = torch.load('%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs))
        decB = torch.load('%s/%s-decB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs))
        decC = torch.load('%s/%s-decC_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs))
    else:
        encA = torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs), map_location='cpu')
        encB = torch.load('%s/%s-encB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs), map_location='cpu')
        encC = torch.load('%s/%s-encC_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs), map_location='cpu')
        decA = torch.load('%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs), map_location='cpu')
        decB = torch.load('%s/%s-decB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs), map_location='cpu')
        decC = torch.load('%s/%s-decC_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs), map_location='cpu')

for e in range(args.ckpt_epochs, args.epochs):
    train_start = time.time()
    train_elbo, rec_lossA, rec_lossB, rec_lossC, train_acc = train(train_data, encA, decA, encB, decB, encC, decC,
                                                                   optimizer)
    train_testset_elbo, rec_lossB_testset, rec_lossC_testset = train_testset(test_data, encB, decB, encC, decC,
                                                                             optimizer)
    train_end = time.time()
    test_start = time.time()
    test_elbo, test_accuracy = test(test_data, encA, decA, encB, decB, e)

    if args.viz_on:
        LINE_GATHER.insert(epoch=e,
                           train_acc=train_acc,
                           test_acc=test_accuracy,
                           test_total_loss=test_elbo,
                           total_loss=train_elbo,
                           recon_A=rec_lossA,
                           recon_B=rec_lossB,
                           recon_C=rec_lossC,
                           rec_lossB_testset=rec_lossB_testset,
                           rec_lossC_testset=rec_lossC_testset
                           )
        visualize_line()
        LINE_GATHER.flush()

    test_end = time.time()
    if (e + 1) % 10 == 0 or e + 1 == args.epochs:
        save_ckpt(e + 1)
        util.evaluation.save_traverse_cub(e, test_data, encA, decA, CUDA, MODEL_NAME, ATTR_DIM,
                                          fixed_idxs=[658, 1570, 2233, 2456, 2880, 1344, 2750, 1800, 1111, 300],
                                          private=False)  # 2880
        util.evaluation.save_traverse_cub(e, train_data, encA, decA, CUDA, MODEL_NAME, ATTR_DIM,
                                          fixed_idxs=[130, 215, 502, 537, 4288, 1000, 2400, 1220, 3002, 3312],
                                          private=False)
    print('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e, Accuracy %0.3f (%ds)' % (
        e, train_elbo, train_end - train_start,
        test_elbo, test_accuracy[0], test_end - test_start))

if args.ckpt_epochs == args.epochs:
    # test_elbo, test_accuracy = test(test_data, encA, decA, encB, decB, 5)
    util.evaluation.save_traverse_cub(args.epochs, test_data, encA, decA, CUDA, MODEL_NAME, ATTR_DIM,
                                      fixed_idxs=[658, 1570, 2233, 2456, 2880, 1344, 2750, 1800, 1111, 300],
                                      private=False)  # 2880
    util.evaluation.save_traverse_cub(args.epochs, train_data, encA, decA, CUDA, MODEL_NAME, ATTR_DIM,
                                      fixed_idxs=[130, 215, 502, 537, 4288, 1000, 2400, 1220, 3002, 3312],
                                      private=False)
    # util.evaluation.mutual_info(test_data, encA, CUDA, flatten_pixel=NUM_PIXELS)
else:
    save_ckpt(args.epochs)
