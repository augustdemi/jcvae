from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import random
import torch
import os
import visdom
import numpy as np

from model import EncoderA, DecoderA, EncoderB2, DecoderB2
from sklearn.metrics import f1_score

import sys

sys.path.append('../')
import probtorch
import util

# ------------------------------------------------
# training parameters

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=6, metavar='N',
                        help='run_id')
    parser.add_argument('--run_desc', type=str, default='',
                        help='run_id desc')
    parser.add_argument('--n_shared', type=int, default=10,
                        help='size of the latent embedding of shared')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--ckpt_epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default: 1e-3]')

    parser.add_argument('--label_frac', type=float, default=1.,
                        help='how many labels to use')
    parser.add_argument('--sup_frac', type=float, default=1.,
                        help='supervision ratio')
    parser.add_argument('--lambda_text', type=float, default=50.,
                        help='multipler for text reconstruction [default: 10]')
    parser.add_argument('--beta1', type=float, default=1.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--beta2', type=float, default=1.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed for get_paired_data')
    parser.add_argument('--wseed', type=int, default=0, metavar='N',
                        help='random seed for weight')

    parser.add_argument('--annealing-epochs', type=int, default=200, metavar='N',
                        help='number of epochs to anneal KL for [default: 200]')
    parser.add_argument('--lamb_annealing_epochs', type=int, default=20, metavar='N',
                        help='number of epochs to anneal KL for [default: 200]')

    parser.add_argument('--ckpt_path', type=str, default='../weights/mnist_mvae/1',
                        help='save and load path for ckpt')

    parser.add_argument('--pretrain',
                        default=False, type=probtorch.util.str2bool, help='enable visdom visualization')

    # visdom
    parser.add_argument('--viz_on',
                        default=False, type=probtorch.util.str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_port',
                        default=8002, type=int, help='visdom port number')

    args = parser.parse_args()

# ------------------------------------------------


EPS = 1e-9
CUDA = torch.cuda.is_available()

# path parameters
MODEL_NAME = 'mnist_mvae_test-run_id%d-shared%02ddim-label_frac%s-sup_frac%s-lamb_text%s-seed%s-bs%s-wseed%s-lr%s-pretrain%s' % (
    args.run_id, args.n_shared, args.label_frac, args.sup_frac, args.lambda_text, args.seed,
    args.batch_size, args.wseed, args.lr, args.pretrain)
DATA_PATH = '../data'

if not os.path.isdir(args.ckpt_path):
    os.makedirs(args.ckpt_path)

if len(args.run_desc) > 1:
    desc_file = os.path.join(args.ckpt_path, 'run_id' + str(args.run_id) + '.txt')
    with open(desc_file, 'w') as outfile:
        outfile.write(args.run_desc)

# model parameters
NUM_PIXELS = 784
TEMP = 0.66


# visdom setup
def viz_init():
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llA'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llB'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['acc'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['total_losses'])


def visualize_line():
    data = LINE_GATHER.data
    recon_A = torch.Tensor(data['recon_A'])
    recon_B = torch.Tensor(data['recon_B'])

    recon_poeA = torch.Tensor(data['recon_poeA'])
    recon_poeB = torch.Tensor(data['recon_poeB'])
    total_loss = torch.Tensor(data['total_loss'])

    epoch = torch.Tensor(data['epoch'])
    test_acc = torch.Tensor(data['test_acc'])
    test_total_loss = torch.Tensor(data['test_total_loss'])

    kl_A = torch.Tensor(data['kl_A'])
    kl_B = torch.Tensor(data['kl_B'])
    kl_poe = torch.Tensor(data['kl_poe'])
    kl = torch.tensor(np.stack([kl_A, kl_B, kl_poe], -1))

    llA = torch.tensor(np.stack([recon_A, recon_poeA], -1))
    llB = torch.tensor(np.stack([recon_B, recon_poeB], -1))
    total_losses = torch.tensor(np.stack([total_loss, test_total_loss], -1))
    acc = torch.tensor(np.stack([test_acc], -1))

    VIZ.line(
        X=epoch, Y=kl, env=MODEL_NAME + '/lines',
        win=WIN_ID['kl'], update='append',
        opts=dict(xlabel='epoch', ylabel='kl-div',
                  title='KL', legend=['kl_A', 'kl_B', 'kl_poe'])
    )

    VIZ.line(
        X=epoch, Y=llA, env=MODEL_NAME + '/lines',
        win=WIN_ID['llA'], update='append',
        opts=dict(xlabel='epoch', ylabel='loglike',
                  title='LL of modalA', legend=['A', 'poeA'])
    )
    VIZ.line(
        X=epoch, Y=llB, env=MODEL_NAME + '/lines',
        win=WIN_ID['llB'], update='append',
        opts=dict(xlabel='epoch', ylabel='loglike',
                  title='LL of modalB', legend=['B', 'poeB'])
    )

    VIZ.line(
        X=epoch, Y=acc, env=MODEL_NAME + '/lines',
        win=WIN_ID['acc'], update='append',
        opts=dict(xlabel='epoch', ylabel='accuracy',
                  title='Accuracy', legend=['test_acc'])
    )

    VIZ.line(
        X=epoch, Y=total_losses, env=MODEL_NAME + '/lines',
        win=WIN_ID['total_losses'], update='append',
        opts=dict(xlabel='epoch', ylabel='loss',
                  title='Total Loss', legend=['train_loss', 'test_loss'])
    )


if args.viz_on:
    WIN_ID = dict(
        llA='win_llA', llB='win_llB', acc='win_acc', total_losses='win_total_losses', kl='win_kl'
    )
    LINE_GATHER = probtorch.util.DataGather(
        'epoch', 'recon_A', 'recon_B', 'recon_poeA', 'recon_poeB',
        'total_loss', 'test_total_loss', 'test_acc', 'kl_A', 'kl_B', 'kl_poe'
    )
    VIZ = visdom.Visdom(port=args.viz_port)
    viz_init()

train_data = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_PATH, train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True)
test_data = torch.utils.data.DataLoader(
    datasets.MNIST(DATA_PATH, train=False, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True)

print('>>> data loaded')
print('train: ', len(train_data.dataset))
print('test: ', len(test_data.dataset))


def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())


encA = EncoderA(args.wseed, zShared_dim=args.n_shared)
decA = DecoderA(args.wseed, zShared_dim=args.n_shared)
encB = EncoderB2(args.wseed, zShared_dim=args.n_shared)
decB = DecoderB2(args.wseed, zShared_dim=args.n_shared)
if CUDA:
    encA.cuda()
    decA.cuda()
    encB.cuda()
    decB.cuda()
    cuda_tensors(encA)
    cuda_tensors(decA)
    cuda_tensors(encB)
    cuda_tensors(decB)

optimizer = torch.optim.Adam(
    list(encB.parameters()) + list(decB.parameters()) + list(encA.parameters()) + list(decA.parameters()),
    lr=args.lr)


#
# def elbo(q, pA, pB, lamb=1.0, annealing_factor=1.0):
#     muA_own = q['sharedA'].dist.loc.squeeze(0)
#     stdA_own = q['sharedA'].dist.scale.squeeze(0)
#     muB_own = q['sharedB'].dist.loc.squeeze(0)
#     stdB_own = q['sharedB'].dist.scale.squeeze(0)
#
#     # from each of modality
#     reconst_loss_A = pA['images_own'].loss.mean()
#     kl_A = 0.5 * torch.sum(1 + torch.log(stdA_own ** 2 + EPS) - muA_own ** 2 - stdA_own ** 2,
#                            dim=1).mean()
#
#     reconst_loss_B = pB['label_own'].loss.mean()
#     kl_B = 0.5 * torch.sum(1 + torch.log(stdB_own ** 2 + EPS) - muB_own ** 2 - stdB_own ** 2,
#                            dim=1).mean()
#
#     if q['poe'] is not None:
#         mu_poe = q['poe'].dist.loc.squeeze(0)
#         std_poe = q['poe'].dist.scale.squeeze(0)
#         reconst_loss_poeA = pA['images_poe'].loss.mean()
#         reconst_loss_poeB = pB['label_poe'].loss.mean()
#         kl_poe = 0.5 * torch.sum(
#             1 + torch.log(std_poe ** 2 + EPS) - mu_poe.pow(2) - std_poe ** 2, dim=1).mean()
#
#         loss = (reconst_loss_A + annealing_factor * kl_A) + (lamb * reconst_loss_B + annealing_factor * kl_B) + (
#             reconst_loss_poeA + lamb * reconst_loss_poeB + annealing_factor * kl_poe)
#
#     else:
#         reconst_loss_poeA = reconst_loss_poeB = kl_poe = None
#         loss = 2 * ((reconst_loss_A + annealing_factor * kl_A) + (lamb * reconst_loss_B + annealing_factor * kl_B))
#
#     return -loss, [reconst_loss_A, reconst_loss_poeA], [reconst_loss_B, reconst_loss_poeB], [kl_A, kl_B, kl_poe]
#



def elbo(q, pA, pB, lamb=1.0, annealing_factor=1.0, lamb_annealing_factor=1.0):
    # from each of modality
    beta1 = (1.0, 1.0, 1.0)
    beta2 = (1.0, 1.0, 1.0)
    bias = 1.0
    reconst_loss_A, kl_A = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_own'], latents=['sharedA'],
                                                               sample_dim=0, batch_dim=1,
                                                               beta=beta1, bias=bias)
    reconst_loss_B, kl_B = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['label_own'], latents=['sharedB'],
                                                               sample_dim=0, batch_dim=1,
                                                               beta=beta2, bias=bias)

    if q['poe'] is not None:
        reconst_loss_poeA, kl_poeA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_poe'], latents=['poe'],
                                                                         sample_dim=0, batch_dim=1,
                                                                         beta=beta1, bias=bias)
        reconst_loss_poeB, kl_poeB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['label_poe'], latents=['poe'],
                                                                         sample_dim=0, batch_dim=1,
                                                                         beta=beta2, bias=bias)

        kl_poe = (kl_poeA + kl_poeB)
        loss = (reconst_loss_A - annealing_factor * kl_A) + (
            lamb_annealing_factor * lamb * reconst_loss_B - annealing_factor * kl_B) + (
                   reconst_loss_poeA + lamb * reconst_loss_poeB - annealing_factor * kl_poe)

    else:
        reconst_loss_poeA = reconst_loss_poeB = kl_poe = None
        loss = 2 * ((reconst_loss_A - annealing_factor * kl_A) + (
            lamb_annealing_factor * lamb * reconst_loss_B - annealing_factor * kl_B))
    return -loss, [reconst_loss_A, reconst_loss_poeA], [reconst_loss_B, reconst_loss_poeB], [kl_A, kl_B, kl_poe]


def train(data, encA, decA, encB, decB, epoch, optimizer,
          label_mask={}, fixed_imgs=None, fixed_labels=None):
    epoch_elbo = 0.0
    epoch_recA = epoch_rec_poeA = 0.0
    epoch_recB = epoch_rec_poeB = 0.0
    kl_A = kl_B = kl_poe = 0.0
    pair_cnt = 0
    encA.train()
    encB.train()
    decA.train()
    decB.train()
    N = 0
    torch.autograd.set_detect_anomaly(True)

    ############I added #########################
    # lamb_annealing_factor = np.power(2.0, (epoch + 1) // args.lamb_annealing_epochs)
    # print('epoch: ', epoch, ', lamb_annealing_factor:', lamb_annealing_factor)
    #####################################


    for b, (images, labels) in enumerate(data):
        if epoch < args.annealing_epochs:
            # compute the KL annealing factor for the current mini-batch in the current epoch
            annealing_factor = (float(b + epoch * len(train_data) + 1) /
                                float(args.annealing_epochs * len(train_data)))
        else:
            # by default the KL annealing factor is unity
            annealing_factor = 1.0

        # annealing_factor = 1
        if args.label_frac > 1 and random.random() < args.sup_frac:
            # print(b)
            N += 1
            shuffled_idx = list(range(int(args.label_frac)))
            random.shuffle(shuffled_idx)
            shuffled_idx = shuffled_idx[:args.batch_size]
            # print(shuffled_idx[:10])
            fixed_imgs_batch = fixed_imgs[shuffled_idx]
            fixed_labels_batch = fixed_labels[shuffled_idx]
            images = fixed_imgs_batch.view(-1, NUM_PIXELS)
            labels_onehot = torch.zeros(args.batch_size, 10)
            labels_onehot.scatter_(1, fixed_labels_batch.unsqueeze(1), 1)
            labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS)
            optimizer.zero_grad()
            if CUDA:
                images = images.cuda()
                labels = labels.cuda()
                labels_onehot = labels_onehot.cuda()

            # encode
            q = encA(images, CUDA)
            q = encB(labels, CUDA, q=q)

            ## poe ##
            mu_poe, std_poe = probtorch.util.apply_poe(CUDA, q['sharedA'].dist.loc, q['sharedA'].dist.scale,
                                                       q['sharedB'].dist.loc, q['sharedB'].dist.scale)
            q.normal(mu_poe,
                     std_poe,
                     name='poe')

            # muA, stdA = probtorch.util.apply_poe(CUDA, q['sharedA'].dist.loc, q['sharedA'].dist.scale)
            # q['sharedA'].dist.loc = muA
            # q['sharedA'].dist.scale = stdA
            #
            # muB, stdB = probtorch.util.apply_poe(CUDA, q['sharedB'].dist.loc, q['sharedB'].dist.scale)
            # q['sharedB'].dist.loc = muB
            # q['sharedB'].dist.scale = stdB

            # decode attr
            shared_dist = {'poe': 'poe', 'own': 'sharedB'}
            pB, _ = decB(labels_onehot, shared_dist, q=q)

            # decode img
            shared_dist = {'poe': 'poe', 'own': 'sharedA'}
            pA = decA(images, shared_dist, q=q)

            for param in encB.parameters():
                param.requires_grad = True
            for param in decB.parameters():
                param.requires_grad = True
            # loss
            loss, recA, recB, kl = elbo(q, pA, pB, lamb=args.lambda_text, annealing_factor=annealing_factor)
        else:
            N += 1
            images = images.view(-1, NUM_PIXELS)
            labels_onehot = torch.zeros(args.batch_size, 10)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS)
            if CUDA:
                images = images.cuda()
                labels = labels.cuda()
                labels_onehot = labels_onehot.cuda()
            optimizer.zero_grad()

            if b not in label_mask:
                label_mask[b] = (random.random() < args.label_frac)

            if (label_mask[b] and args.label_frac == args.sup_frac):
                # encode
                q = encA(images, CUDA)
                q = encB(labels, CUDA, q=q)

                ## poe ##
                mu_poe, std_poe = probtorch.util.apply_poe(CUDA, q['sharedA'].dist.loc, q['sharedA'].dist.scale,
                                                           q['sharedB'].dist.loc, q['sharedB'].dist.scale)
                q.normal(mu_poe,
                         std_poe,
                         name='poe')

                # muA, stdA = probtorch.util.apply_poe(CUDA, q['sharedA'].dist.loc, q['sharedA'].dist.scale)
                # muB, stdB = probtorch.util.apply_poe(CUDA, q['sharedB'].dist.loc, q['sharedB'].dist.scale)
                # q['sharedA'].dist.loc = muA
                # q['sharedA'].dist.scale = stdA
                # q['sharedB'].dist.loc = muB
                # q['sharedB'].dist.scale = stdB

                # muA, stdA = probtorch.util.apply_poe(CUDA, q['sharedA'].dist.loc, q['sharedA'].dist.scale)
                # muB, stdB = probtorch.util.apply_poe(CUDA, q['sharedB'].dist.loc, q['sharedB'].dist.scale)
                # mu, logvar = probtorch.util.prior_expert((1, args.batch_size, args.n_shared),
                #                           use_cuda=CUDA)
                #
                #
                # mu = torch.cat((mu, muA), dim=0)
                # logvar = torch.cat((logvar, torch.log(stdA ** 2 + EPS)), dim=0)
                #
                # mu = torch.cat((mu, muB), dim=0)
                # logvar = torch.cat((logvar, torch.log(stdB ** 2 + EPS)), dim=0)

                # decode attr
                shared_dist = {'poe': 'poe', 'own': 'sharedB'}
                pB, _ = decB(labels_onehot, shared_dist, q=q)

                # decode img
                shared_dist = {'poe': 'poe', 'own': 'sharedA'}
                pA = decA(images, shared_dist, q=q)

                for param in encB.parameters():
                    param.requires_grad = True
                for param in decB.parameters():
                    param.requires_grad = True
                # loss
                loss, recA, recB, kl = elbo(q, pA, pB, lamb=args.lambda_text, annealing_factor=annealing_factor)
            else:
                shuffled_idx = list(range(args.batch_size))
                random.shuffle(shuffled_idx)
                labels_onehot = labels_onehot[shuffled_idx]

                # encode
                q = encA(images, CUDA)
                q = encB(labels, CUDA, q=q)

                # decode attr
                shared_dist = {'own': 'sharedB'}
                pB, _ = decB(labels_onehot, shared_dist, q=q)

                # muA, stdA = probtorch.util.apply_poe(CUDA, q['sharedA'].dist.loc, q['sharedA'].dist.scale)
                # q['sharedA'].dist.loc = muA
                # q['sharedA'].dist.scale = stdA
                #
                # muB, stdB = probtorch.util.apply_poe(CUDA, q['sharedB'].dist.loc, q['sharedB'].dist.scale)
                # q['sharedB'].dist.loc = muB
                # q['sharedB'].dist.scale = stdB

                # decode img
                shared_dist = {'own': 'sharedA'}
                pA = decA(images, shared_dist, q=q)
                for param in encB.parameters():
                    param.requires_grad = False
                for param in decB.parameters():
                    param.requires_grad = False
                loss, recA, recB, kl = elbo(q, pA, pB, lamb=args.lambda_text, annealing_factor=annealing_factor)

        loss.backward()
        optimizer.step()
        if CUDA:
            loss = loss.cpu()
            recA[0] = recA[0].cpu()
            recB[0] = recB[0].cpu()
            kl[0] = kl[0].cpu()
            kl[1] = kl[1].cpu()

        epoch_elbo += loss.item()
        epoch_recA += recA[0].item()
        epoch_recB += recB[0].item()
        kl_A += kl[0].item()
        kl_B += kl[1].item()

        if recA[1] is not None:
            if CUDA:
                kl[2] = kl[2].cpu()
                for i in range(2):
                    recA[i] = recA[i].cpu()
                    recB[i] = recB[i].cpu()
            epoch_rec_poeA += recA[1].item()
            epoch_rec_poeB += recB[1].item()
            kl_poe += kl[2].item()
            pair_cnt += 1

        if b % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)], annealing_factor: {:.3f})'.format(
                e, b * args.batch_size, len(data.dataset),
                   100. * b * args.batch_size / len(data.dataset), annealing_factor))
    return epoch_elbo / N, [epoch_recA / N, epoch_rec_poeA / pair_cnt], [epoch_recB / N,
                                                                         epoch_rec_poeB / pair_cnt], [kl_A / N,
                                                                                                      kl_B / N,
                                                                                                      kl_poe / pair_cnt], label_mask


def test(data, encA, decA, encB, decB):
    encA.eval()
    decA.eval()
    encB.eval()
    decB.eval()
    epoch_elbo = 0.0
    epoch_acc = 0
    N = 0
    for b, (images, labels) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += 1
            images = images.view(-1, NUM_PIXELS)
            labels_onehot = torch.zeros(args.batch_size, 10)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS)
            if CUDA:
                images = images.cuda()
                labels = labels.cuda()
                labels_onehot = labels_onehot.cuda()

            # encode
            q = encA(images, CUDA)
            q = encB(labels, CUDA, q=q)

            # muA, stdA = probtorch.util.apply_poe(CUDA, q['sharedA'].dist.loc, q['sharedA'].dist.scale)
            # q['sharedA'].dist.loc = muA
            # q['sharedA'].dist.scale = stdA
            #
            # muB, stdB = probtorch.util.apply_poe(CUDA, q['sharedB'].dist.loc, q['sharedB'].dist.scale)
            # q['sharedB'].dist.loc = muB
            # q['sharedB'].dist.scale = stdB

            # decode attr
            shared_dist = {'own': 'sharedB', 'cross': 'sharedA'}
            pB, pred_labels = decB(labels_onehot, shared_dist, q=q, train=False)

            # decode img
            shared_dist = {'own': 'sharedA', 'cross': 'sharedB'}
            pA = decA(images, shared_dist, q=q)

            batch_elbo, _, _, _ = elbo(q, pA, pB, lamb=args.lambda_text)

            if CUDA:
                batch_elbo = batch_elbo.cpu()
                pred_labels = pred_labels.cpu()
                labels = labels.cpu()
            epoch_elbo += batch_elbo.item()

            pred = pred_labels.max(-1)[1]
            pred = pred.detach().numpy()
            target = labels.detach().numpy()
            epoch_acc += (pred == target).mean()

    return epoch_elbo / N, epoch_acc / N


def save_ckpt(e):
    if not os.path.isdir(args.ckpt_path):
        os.mkdir(args.ckpt_path)
    torch.save(encA.state_dict(),
               '%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(decA.state_dict(),
               '%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(encB.state_dict(),
               '%s/%s-encB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))
    torch.save(decB.state_dict(),
               '%s/%s-decB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))


def get_paired_data(paired_cnt, seed):
    data = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_PATH, train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=False)
    tr_labels = data.dataset.targets

    cnt = int(paired_cnt / 10)
    assert cnt == paired_cnt / 10

    label_idx = {}
    for i in range(10):
        label_idx.update({i: []})
    for idx in range(len(tr_labels)):
        label = int(tr_labels[idx].data.detach().cpu().numpy())
        label_idx[label].append(idx)

    total_random_idx = []
    for i in range(10):
        random.seed(seed)
        per_label_random_idx = random.sample(label_idx[i], cnt)
        total_random_idx.extend(per_label_random_idx)
    random.seed(seed)
    random.shuffle(total_random_idx)

    imgs = []
    labels = []
    for idx in total_random_idx:
        img, label = data.dataset.__getitem__(idx)
        imgs.append(img)
        labels.append(torch.tensor(label))
    imgs = torch.stack(imgs, dim=0)
    labels = torch.stack(labels, dim=0)
    return imgs, labels


if args.ckpt_epochs > 0:
    if CUDA:
        encA.load_state_dict(torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs)))
        decA.load_state_dict(torch.load('%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs)))
        encB.load_state_dict(torch.load('%s/%s-encB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs)))
        decB.load_state_dict(torch.load('%s/%s-decB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs)))
    else:
        encA.load_state_dict(torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
                                        map_location=torch.device('cpu')))
        decA.load_state_dict(torch.load('%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
                                        map_location=torch.device('cpu')))
        encB.load_state_dict(torch.load('%s/%s-encB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
                                        map_location=torch.device('cpu')))
        decB.load_state_dict(torch.load('%s/%s-decB_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
                                        map_location=torch.device('cpu')))

if args.pretrain:
    print('>>>> pretrained encA')
    pretrain_model = '../weights/mnist_mvae_pretrain/mnist_mvae_pretrain-run_id1-shared10-bs100-lr0.001-encA_epoch100.rar'
if CUDA:
    encA.load_state_dict(torch.load(pretrain_model))
else:
    encA.load_state_dict(torch.load(pretrain_model, map_location=torch.device('cpu')))

mask = {}
fixed_imgs = None
fixed_labels = None
if args.label_frac > 1:
    fixed_imgs, fixed_labels = get_paired_data(args.label_frac, args.seed)

for e in range(args.ckpt_epochs, args.epochs):
    train_start = time.time()
    train_elbo, rec_lossA, rec_lossB, kl, mask = train(train_data, encA, decA, encB, decB, e,
                                                       optimizer, mask, fixed_imgs=fixed_imgs,
                                                       fixed_labels=fixed_labels)
    train_end = time.time()

    test_start = time.time()
    test_elbo, test_accuracy = test(test_data, encA, decA, encB, decB)
    test_end = time.time()

    if args.viz_on:
        LINE_GATHER.insert(epoch=e,
                           test_acc=test_accuracy,
                           test_total_loss=test_elbo,
                           total_loss=train_elbo,
                           recon_A=rec_lossA[0],
                           recon_poeA=rec_lossA[1],
                           recon_B=rec_lossB[0],
                           recon_poeB=rec_lossB[1],
                           kl_A=kl[0],
                           kl_B=kl[1],
                           kl_poe=kl[2]
                           )
        visualize_line()
        LINE_GATHER.flush()

    if (e + 1) % 20 == 0 or e + 1 == args.epochs:
        save_ckpt(e + 1)
    print(
        '[Epoch %d] Train: ELBO %.4e (%ds), Test: ELBO %.4e, Accuracy %0.3f (%ds)' % (
            e, train_elbo, train_end - train_start,
            test_elbo, test_accuracy, test_end - test_start))

if args.ckpt_epochs == args.epochs:
    decA.eval()
    encB.eval()

    util.evaluation.save_cross_mnist_mvae(args.ckpt_epochs, decA, encB, 64,
                                          CUDA, MODEL_NAME)

    test_elbo, test_accuracy = test(test_data, encA, decA, encB, decB)

    print('test_accuracy:', test_accuracy)

else:
    save_ckpt(args.epochs)


####
def visualize_line_metrics(self, iters, metric1, metric2):
    # prepare data to plot
    iters = torch.tensor([iters], dtype=torch.int64).detach()
    metric1 = torch.tensor([metric1])
    metric2 = torch.tensor([metric2])
    metrics = torch.stack([metric1.detach(), metric2.detach()], -1)

    VIZ.line(
        X=iters, Y=metrics, env=MODEL_NAME + '/lines',
        win=WIN_ID['metrics'], update='append',
        opts=dict(xlabel='iter', ylabel='metrics',
                  title='Disentanglement metrics',
                  legend=['metric1', 'metric2'])
    )
