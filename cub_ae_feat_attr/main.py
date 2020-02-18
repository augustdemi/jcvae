from torchvision import datasets, transforms

import time
import random
import torch
import os
import numpy as np
from model import EncoderImgF, DecoderImgF, EncoderAttr, DecoderAttr, EncoderA, DecoderA2
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
    parser.add_argument('--ckpt_epochs', type=int, default=0, metavar='N',
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
MODEL_NAME = 'cub_ae_feat_attr-run_id%d-privA%02ddim-shared%02ddim-lamb%s-beta%s-lr%s-bs%s-wseed%s-seed%s-num_hidden%s' % (
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

    ae_train_loss = torch.Tensor(data['ae_train_loss'])
    ae_test_loss = torch.Tensor(data['ae_test_loss'])

    total_losses = torch.tensor(np.stack([total_loss, test_total_loss, val_total_loss], -1))
    ae_losses = torch.tensor(np.stack([ae_train_loss, ae_test_loss], -1))
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
        opts=dict(xlabel='epoch', ylabel='distance',
                  title='W. Dist.', legend=['tr_acc', 'val_acc', 'te_acc'])
    )

    VIZ.line(
        X=epoch, Y=ae_losses, env=MODEL_NAME + '/lines',
        win=WIN_ID['ae_losses'], update='append',
        opts=dict(xlabel='epoch', ylabel='loss',
                  title='AE Loss', legend=['ae_train_loss', 'ae_test_loss'])
    )


if args.viz_on:
    WIN_ID = dict(
        llA='win_llA', llB='win_llB',
        total_losses='win_total_losses',
        llB_test='win_llB_test', llA_test='win_llA_test',
        llB_val='win_llB_val', llA_val='win_llA_val', acc='win_acc', ae_losses='win_ae_losses',
    )
    LINE_GATHER = probtorch.util.DataGather(
        'epoch', 'recon_A', 'recon_B',
        'total_loss', 'test_total_loss', 'recon_A_test', 'recon_B_test',
        'val_total_loss', 'recon_A_val', 'recon_B_val', 'tr_acc', 'te_acc', 'val_acc', 'ae_train_loss', 'ae_test_loss'
    )
    VIZ = visdom.Visdom(port=args.viz_port)
    viz_init()

train_data = torch.utils.data.DataLoader(datasets(path, train=True, crop=1.2), batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=len(GPU))

test_data = torch.utils.data.DataLoader(datasets(path, train=False, crop=1.2), batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=len(GPU))

val_data = torch.utils.data.DataLoader(datasets(path, train=True, crop=1.2, val=True), batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=len(GPU))
BIAS_TRAIN = (train_data.dataset.__len__() - 1) / (args.batch_size - 1)
BIAS_TEST = (test_data.dataset.__len__() - 1) / (args.batch_size - 1)


def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())


encA = EncoderImgF(args.wseed, zPrivate_dim=args.n_privateA, zShared_dim=args.n_shared, num_hidden=args.num_hidden)
decA = DecoderImgF(args.wseed, zPrivate_dim=args.n_privateA, zShared_dim=args.n_shared, num_hidden=args.num_hidden)
encB = EncoderAttr(args.wseed, zShared_dim=args.n_shared, num_hidden=args.num_hidden)
decB = DecoderAttr(args.wseed, zShared_dim=args.n_shared, num_hidden=args.num_hidden)

ae_encA = EncoderA(0)
ae_decA = DecoderA2(0)


if CUDA:
    encA.cuda()
    decA.cuda()
    encB.cuda()
    decB.cuda()
    ae_encA.cuda()
    ae_decA.cuda()
    cuda_tensors(encA)
    cuda_tensors(decA)
    cuda_tensors(encB)
    cuda_tensors(decB)
    cuda_tensors(ae_encA)
    cuda_tensors(ae_decA)

if args.ckpt_epochs > 0:
    if CUDA:
        encA = torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs))
        encB = torch.load('%s/%s-encB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs))
        decA = torch.load('%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs))
        decB = torch.load('%s/%s-decB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs))
        ae_encA = torch.load('%s/%s-ae_encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs))
        ae_decA = torch.load('%s/%s-ae_decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs))
    else:
        encA = torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs), map_location='cpu')
        encB = torch.load('%s/%s-encB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs), map_location='cpu')
        decA = torch.load('%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs), map_location='cpu')
        decB = torch.load('%s/%s-decB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs), map_location='cpu')
        ae_encA = torch.load('%s/%s-ae_encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
                             map_location='cpu')
        ae_decA = torch.load('%s/%s-ae_decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
                             map_location='cpu')

if CUDA:
    ae_encA = torch.load('../weights/cub_img_ae/cub-img-ae-run_id3-bs64-encA_epoch200.rar')
    ae_decA = torch.load('../weights/cub_img_ae/cub-img-ae-run_id3-bs64-decA_epoch200.rar')
else:
    ae_encA = torch.load('../weights/cub_img_ae/cub-img-ae-run_id3-bs64-encA_epoch200.rar', map_location='cpu')
    ae_decA = torch.load('../weights/cub_img_ae/cub-img-ae-run_id3-bs64-decA_epoch200.rar', map_location='cpu')


ae_optimizer = torch.optim.Adam(
    list(ae_decA.parameters()),
    lr=args.lr)

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


def train(data, encA, decA, encB, decB, ae_enc, ae_dec, optimizer, ae_optimizer):
    epoch_elbo = epoch_distance = epoch_ae_loss = 0.0
    epoch_recA = epoch_rec_poeA = epoch_rec_crA = 0.0
    epoch_recB = epoch_rec_poeB = epoch_rec_crB = 0.0

    ae_enc.eval()
    N = 0
    for b, (images, attributes, _) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += 1

            encA.train()
            encB.train()
            decA.train()
            decB.train()
            ae_dec.eval()

            optimizer.zero_grad()
            if CUDA:
                images = images.cuda()
                attributes = attributes.cuda()
            # encode
            img_feat = ae_enc(images)
            q = encA(img_feat, num_samples=NUM_SAMPLES)
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
            pA, recon_feat = decA(img_feat, shared_dist, q=q, num_samples=NUM_SAMPLES)

            # loss
            loss, recA, recB = elbo(q, pA, pB, lamb=lamb, beta=beta, bias=BIAS_TRAIN)
            loss.backward(retain_graph=True)
            optimizer.step()

            ######################################
            encA.eval()
            encB.eval()
            decA.eval()
            decB.eval()
            ae_dec.train()
            ae_optimizer.zero_grad()

            # original feat
            recon_images = ae_dec(img_feat)
            recon_images = recon_images.view(recon_images.size(0), -1)

            # recon img modal feat
            recon_images_own = ae_dec(recon_feat['own'])
            recon_images_own = recon_images_own.view(recon_images_own.size(0), -1)

            # recon cross modal feat
            recon_images_cross = ae_dec(recon_feat['cross'])
            recon_images_cross = recon_images_cross.view(recon_images_cross.size(0), -1)

            images = images.view(images.size(0), -1)
            ae_loss = recon_loss(recon_images, images).mean() + recon_loss(recon_images_own,
                                                                           images).mean() + recon_loss(
                recon_images_cross, images).mean()
            ae_loss.backward()
            ae_optimizer.step()
            ######################################

            if CUDA:
                loss = loss.cpu()
                ae_loss = ae_loss.cpu()
                for i in range(3):
                    recA[i] = recA[i].cpu()
                    recB[i] = recB[i].cpu()

            epoch_elbo += loss.item()
            epoch_ae_loss += ae_loss.item() / 3

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
                distance = distance.cpu()


            epoch_distance += distance.item()

    print('enc:', ae_enc.resnet[0].weight.sum())
    print('dec:', ae_dec.dec_image[0].weight.sum())
    return epoch_elbo / N, [epoch_recA / N, epoch_rec_poeA / N, epoch_rec_crA / N], \
           [epoch_recB / N, epoch_rec_poeB / N, epoch_rec_crB / N], epoch_distance / N, epoch_ae_loss / N


def test(data, encA, decA, encB, decB, ae_enc, ae_dec):
    epoch_elbo = epoch_distance = epoch_ae_loss = 0.0
    epoch_recA = epoch_rec_crA = 0.0
    epoch_recB = epoch_rec_crB = 0.0
    encA.eval()
    encB.eval()
    decA.eval()
    decB.eval()
    ae_enc.eval()
    ae_dec.eval()

    N = 0
    for b, (images, attributes, _) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += 1
            if CUDA:
                images = images.cuda()
                attributes = attributes.cuda()
            # encode
            img_feat = ae_enc(images)
            q = encA(img_feat, num_samples=NUM_SAMPLES)
            q = encB(attributes, num_samples=NUM_SAMPLES, q=q)

            # decode attr
            shared_dist = {'cross': 'sharedA', 'own': 'sharedB'}
            pB = decB(attributes, shared_dist, q=q, num_samples=NUM_SAMPLES)

            # decode img
            shared_dist = {'cross': 'sharedB', 'own': 'sharedA'}
            pA, recon_feat = decA(img_feat, shared_dist, q=q, num_samples=NUM_SAMPLES)

            # loss
            loss, recA, recB = elbo(q, pA, pB, lamb=lamb, beta=beta, bias=BIAS_TEST, train=False)

            # original feat
            recon_images = ae_dec(img_feat)
            recon_images = recon_images.view(recon_images.size(0), -1)

            # recon img modal feat
            recon_images_own = ae_dec(recon_feat['own'])
            recon_images_own = recon_images_own.view(recon_images_own.size(0), -1)

            # recon cross modal feat
            recon_images_cross = ae_dec(recon_feat['cross'])
            recon_images_cross = recon_images_cross.view(recon_images_cross.size(0), -1)

            images = images.view(images.size(0), -1)
            ae_loss = recon_loss(recon_images, images).mean() + recon_loss(recon_images_own,
                                                                           images).mean() + recon_loss(
                recon_images_cross, images).mean()

            if CUDA:
                loss = loss.cpu()
                ae_loss = ae_loss.cpu()
                for i in [0, 2]:
                    recA[i] = recA[i].cpu()
                    recB[i] = recB[i].cpu()

            epoch_elbo += loss.item()
            epoch_ae_loss += ae_loss.item() / 3
            epoch_recA += recA[0].item()
            epoch_rec_crA += recA[2].item()

            epoch_recB += recB[0].item()
            epoch_rec_crB += recB[2].item()

            distance = torch.sqrt(torch.sum((q['sharedA'].dist.loc - q['sharedB'].dist.loc) ** 2, dim=1) + \
                                  torch.sum((q['sharedA'].dist.scale - q['sharedB'].dist.scale) ** 2, dim=1))

            distance = distance.sum()
            if CUDA:
                distance = distance.cpu()

            epoch_distance += distance.item()

    return epoch_elbo / N, [epoch_recA / N, epoch_rec_crA / N], \
           [epoch_recB / N, epoch_rec_crB / N], epoch_distance / N, epoch_ae_loss / N


##################### AE ###########################
def recon_loss(recon, orig):
    return -(torch.log(recon + EPS) * orig +
             torch.log(1 - recon + EPS) * (1 - orig)).sum(-1)


def train_ae(data, ae_enc, ae_dec, ae_optimizer):
    epoch_loss = 0.0
    ae_enc.train()
    ae_dec.train()
    N = 0

    for b, (images, _, _) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += 1
            ae_optimizer.zero_grad()
            if CUDA:
                images = images.cuda()
            feature = ae_enc(images)
            recon_images = ae_dec(feature)
            recon_images = recon_images.view(recon_images.size(0), -1)
            images = images.view(images.size(0), -1)
            loss = recon_loss(recon_images, images).mean()
            loss.backward()
            ae_optimizer.step()
            if CUDA:
                loss = loss.cpu()
            epoch_loss += loss.item()
    return epoch_loss / N


def test_ae(data, encA, decA):
    encA.eval()
    decA.eval()
    epoch_loss = 0.0
    N = 0
    for b, (images, _, _) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += 1
            if CUDA:
                images = images.cuda()
            feature = encA(images)
            recon_images = decA(feature)
            recon_images = recon_images.view(recon_images.size(0), -1)
            images = images.view(images.size(0), -1)
            loss = recon_loss(recon_images, images).mean()
            if CUDA:
                loss = loss.cpu()
            epoch_loss += loss.item()
    return epoch_loss / N


####
def save_ckpt(e):
    torch.save(encA, '%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(encB, '%s/%s-encB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(decA, '%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(decB, '%s/%s-decB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(ae_encA, '%s/%s-ae_enc_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(ae_decA, '%s/%s-ae_dec_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def recon(encA, encB, decA, ae_encA, ae_decA, e):
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

    img_feat = ae_encA(imgs)
    q = encA(img_feat, num_samples=1)
    q = encB(attributes, num_samples=NUM_SAMPLES, q=q)

    latents_own = [q['privateA'].dist.loc, q['sharedA'].dist.loc]
    latents_cross = [q['privateA'].dist.loc, q['sharedB'].dist.loc]

    recon_img_feat_own = decA.forward2(latents_own)
    recon_img_own = ae_decA.forward(recon_img_feat_own)

    recon_img_feat_cross = decA.forward2(latents_cross)
    recon_img_cross = ae_decA.forward(recon_img_feat_cross)

    recon_img = ae_decA.forward(img_feat)

    from torchvision.utils import save_image

    out_dir = os.path.join('../output/' + MODEL_NAME, str(e), str(fixed_idxs))
    mkdirs(out_dir)
    # save_image(imgs,
    #            str(os.path.join(out_dir, 'gt_image.png')), nrow=int(np.sqrt(imgs.shape[0])))
    save_image(recon_img_own,
               str(os.path.join(out_dir, 'recon_img_own.png')), nrow=int(np.sqrt(recon_img_own.shape[0])))
    save_image(recon_img_cross,
               str(os.path.join(out_dir, 'recon_img_cross.png')), nrow=int(np.sqrt(recon_img_cross.shape[0])))
    save_image(recon_img,
               str(os.path.join(out_dir, 'recon_img.png')), nrow=int(np.sqrt(recon_img.shape[0])))


for e in range(args.ckpt_epochs, args.epochs):
    train_start = time.time()
    train_elbo, rec_lossA, rec_lossB, tr_dist, ae_train_loss = train(train_data, encA, decA, encB, decB, ae_encA,
                                                                     ae_decA, optimizer, ae_optimizer)
    # ae_train_loss = train_ae(train_data, ae_encA, ae_decA, ae_optimizer)
    train_end = time.time()

    val_start = time.time()
    val_elbo, recon_A_val, recon_B_val, val_dist, _ = test(val_data, encA, decA, encB, decB, ae_encA, ae_decA)
    val_end = time.time()

    test_start = time.time()
    test_elbo, recon_A_test, recon_B_test, te_dist, ae_test_loss = test(test_data, encA, decA, encB, decB, ae_encA,
                                                                        ae_decA)
    test_end = time.time()

    # ae_test_loss = test_ae(test_data, ae_encA, ae_decA)

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
                           te_acc=te_dist,
                           ae_train_loss=ae_train_loss,
                           ae_test_loss=ae_test_loss
                           )
        visualize_line()
        LINE_GATHER.flush()

    if (e + 1) % 10 == 0 or e + 1 == args.epochs:
        save_ckpt(e + 1)
        recon(encA, encB, decA, ae_encA, ae_decA, e)
        # util.evaluation.save_traverse_cub_ia2(e, test_data, encA, decA, CUDA, MODEL_NAME, ATTR_DIM,
        #                                       fixed_idxs=[277, 342, 658, 1570, 2233, 2388, 2880, 1344, 2750, 1111],
        #                                       private=False)  # 2880
        # util.evaluation.save_traverse_cub_ia2(e, train_data, encA, decA, CUDA, MODEL_NAME, ATTR_DIM,
        #                                       fixed_idxs=[336, 502, 537, 575, 4288, 1000, 2400, 1220, 3002, 3312],
        #                                       private=False)
    print('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e, cross_attr %0.3f (%ds)' % (
        e, train_elbo, train_end - train_start,
        test_elbo, recon_B_test[1], test_end - test_start))

if args.ckpt_epochs == args.epochs:

    if CUDA:
        ae_decA = torch.load('../weights/cub_img_ae/cub-img-ae-run_id3-bs64-decA_epoch200.rar')
    else:
        ae_decA = torch.load('../weights/cub_img_ae/cub-img-ae-run_id3-bs64-decA_epoch200.rar', map_location='cpu')


else:
    save_ckpt(args.epochs)
