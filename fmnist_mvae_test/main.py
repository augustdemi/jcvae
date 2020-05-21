from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import torchvision.datasets as dset
import random
import torch
import os
import visdom
import numpy as np
from datasets import DIGIT
from model import EncoderA, EncoderB, DecoderA, DecoderB

import sys

sys.path.append('../')
import probtorch
import util

# ------------------------------------------------
# training parameters

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=42, metavar='N',
                        help='run_id')
    parser.add_argument('--run_desc', type=str, default='',
                        help='run_id desc')
    parser.add_argument('--n_shared', type=int, default=10,
                        help='size of the latent embedding of shared')
    parser.add_argument('--n_private', type=int, default=10,
                        help='size of the latent embedding of private')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--ckpt_epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default: 1e-3]')

    parser.add_argument('--label_frac', type=float, default=1.,
                        help='how many labels to use')
    parser.add_argument('--sup_frac', type=float, default=1.,
                        help='supervision ratio')
    parser.add_argument('--lambda_text', type=float, default=1000.,
                        help='multipler for text reconstruction [default: 10]')
    parser.add_argument('--beta1', type=float, default=3.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--beta2', type=float, default=1.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed for get_paired_data')
    parser.add_argument('--wseed', type=int, default=0, metavar='N',
                        help='random seed for weight')

    parser.add_argument('--ckpt_path', type=str, default='../weights/fmnist/',
                        help='save and load path for ckpt')

    parser.add_argument('--annealing-epochs', type=int, default=200, metavar='N',
                        help='number of epochs to anneal KL for [default: 200]')

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
MODEL_NAME = 'fmnist_mvae_test-run_id%d-priv%02ddim-label_frac%s-sup_frac%s-lamb_text%s-beta1%s-beta2%s-seed%s-bs%s-wseed%s' % (
    args.run_id, args.n_private, args.label_frac, args.sup_frac, args.lambda_text, args.beta1, args.beta2, args.seed,
    args.batch_size, args.wseed)
DATA_PATH = '../data'

if not os.path.isdir(args.ckpt_path):
    os.makedirs(args.ckpt_path)

if len(args.run_desc) > 1:
    desc_file = os.path.join(args.ckpt_path, 'run_id' + str(args.run_id) + '.txt')
    with open(desc_file, 'w') as outfile:
        outfile.write(args.run_desc)

BETA1 = (1., args.beta1, 1.)
BETA2 = (1., args.beta2, 1.)
# model parameters
NUM_PIXELS = 784
TEMP = 0.66
NUM_SAMPLES = 1


# visdom setup
def viz_init():
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llA'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llB'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['test_acc'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['tr_acc'])
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
    tr_acc = torch.Tensor(data['tr_acc'])
    test_total_loss = torch.Tensor(data['test_total_loss'])

    llA = torch.tensor(np.stack([recon_A, recon_poeA], -1))
    llB = torch.tensor(np.stack([recon_B, recon_poeB], -1))
    total_losses = torch.tensor(np.stack([total_loss, test_total_loss], -1))

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
        X=epoch, Y=test_acc, env=MODEL_NAME + '/lines',
        win=WIN_ID['test_acc'], update='append',
        opts=dict(xlabel='epoch', ylabel='accuracy',
                  title='Test Accuracy', legend=['acc'])
    )

    VIZ.line(
        X=epoch, Y=tr_acc, env=MODEL_NAME + '/lines',
        win=WIN_ID['tr_acc'], update='append',
        opts=dict(xlabel='epoch', ylabel='accuracy',
                  title='Training Accuracy', legend=['acc'])
    )

    VIZ.line(
        X=epoch, Y=total_losses, env=MODEL_NAME + '/lines',
        win=WIN_ID['total_losses'], update='append',
        opts=dict(xlabel='epoch', ylabel='loss',
                  title='Total Loss', legend=['train_loss', 'test_loss'])
    )


if args.viz_on:
    WIN_ID = dict(
        llA='win_llA', llB='win_llB', test_acc='win_test_acc', total_losses='win_total_losses', tr_acc='win_tr_acc'
    )
    LINE_GATHER = probtorch.util.DataGather(
        'epoch', 'recon_A', 'recon_B', 'recon_poeA', 'recon_poeB',
        'total_loss', 'test_total_loss', 'test_acc', 'tr_acc'
    )
    VIZ = visdom.Visdom(port=args.viz_port)
    viz_init()

train_data = torch.utils.data.DataLoader(DIGIT('./data', train=True), batch_size=args.batch_size, shuffle=False)
test_data = torch.utils.data.DataLoader(DIGIT('./data', train=False), batch_size=args.batch_size, shuffle=False)

# train_data = torch.utils.data.DataLoader(
#     dset.FashionMNIST('../../data/fMNIST', train=True, download=True,
#                       transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=False)
# test_data = torch.utils.data.DataLoader(
#     dset.FashionMNIST('../../data/fMNIST', train=False, download=True,
#                       transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=False)


train_data_size = len(train_data)

BIAS_TRAIN = (train_data_size - 1) / (args.batch_size - 1)
BIAS_TEST = (test_data.dataset.__len__() - 1) / (args.batch_size - 1)


def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())


encA = EncoderA(args.wseed, zShared_dim=args.n_shared)
decA = DecoderA(args.wseed, zShared_dim=args.n_shared)
encB = EncoderB(args.wseed, zShared_dim=args.n_shared)
decB = DecoderB(args.wseed, zShared_dim=args.n_shared)
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


def elbo(q, pA, pB, annealing_factor, lamb=1.0, beta1=(1.0, 1.0, 1.0), beta2=(1.0, 1.0, 1.0), bias=1.0):
    # from each of modality
    reconst_loss_A, kl_A = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_sharedA'],
                                                               latents=['privateA', 'sharedA'], sample_dim=0,
                                                               batch_dim=1,
                                                               beta=beta1, bias=bias)
    reconst_loss_B, kl_B = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['labels_sharedB'], latents=['sharedB'],
                                                               sample_dim=0, batch_dim=1,
                                                               beta=beta2, bias=bias)

    if q['poe'] is not None:
        reconst_loss_poeA, kl_poeA = probtorch.objectives.mws_tcvae.elbo(q, pA, pA['images_poe'],
                                                                         latents=['privateA', 'poe'], sample_dim=0,
                                                                         batch_dim=1,
                                                                         beta=beta1, bias=bias)
        reconst_loss_poeB, kl_poeB = probtorch.objectives.mws_tcvae.elbo(q, pB, pB['labels_poe'], latents=['poe'],
                                                                         sample_dim=0, batch_dim=1,
                                                                         beta=beta2, bias=bias)

        loss = (reconst_loss_A - annealing_factor * kl_A) + (lamb * reconst_loss_B - annealing_factor * kl_B) + \
               (reconst_loss_poeA - annealing_factor * kl_poeA) + (
                   lamb * reconst_loss_poeB - annealing_factor * kl_poeB)
    else:
        reconst_loss_poeA = reconst_loss_crA = reconst_loss_poeB = reconst_loss_crB = None
        loss = 2 * (reconst_loss_A - annealing_factor * kl_A)
    return -loss, [reconst_loss_A, reconst_loss_poeA], [reconst_loss_B, reconst_loss_poeB]


def train(data, encA, decA, encB, decB, optimizer,
          label_mask, epoch):
    epoch_elbo = 0.0
    epoch_recA = epoch_rec_poeA = epoch_rec_crA = 0.0
    epoch_recB = epoch_rec_poeB = epoch_rec_crB = 0.0
    pair_cnt = 0
    encA.train()
    encB.train()
    decA.train()
    decB.train()
    N = 0
    cnt = 0
    epoch_correct = 0
    torch.autograd.set_detect_anomaly(True)
    for b, (images, labels) in enumerate(data):
        N += 1

        if epoch < args.annealing_epochs:
            # compute the KL annealing factor for the current mini-batch in the current epoch
            annealing_factor = (float(b + epoch * len(train_data) + 1) /
                                float(args.annealing_epochs * len(train_data)))
            # annealing_factor = 1.0
        else:
            # by default the KL annealing factor is unity
            annealing_factor = 1.0

        # images = images.view(-1, NUM_PIXELS)
        labels_onehot = torch.zeros(args.batch_size, args.n_shared)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS)
        if CUDA:
            images = images.cuda()
            labels_onehot = labels_onehot.cuda()
        optimizer.zero_grad()

        if label_mask[b]:
            cnt += 1
            # encode
            # print(images.sum())
            q = encA(images, num_samples=NUM_SAMPLES)
            q = encB(labels_onehot, num_samples=NUM_SAMPLES, q=q)
            ## poe ##
            mu_poe, std_poe = probtorch.util.apply_poe(CUDA, q['sharedA'].dist.loc, q['sharedA'].dist.scale,
                                                       q['sharedB'].dist.loc, q['sharedB'].dist.scale)
            q.normal(mu_poe,
                     std_poe,
                     name='poe')
            # decode attr
            pB = decB(labels_onehot, {'sharedB': q['sharedB'], 'poe': q['poe']}, q=q, num_samples=NUM_SAMPLES)

            # decode img
            pA = decA(images, {'sharedA': q['sharedA'], 'poe': q['poe']}, q=q, num_samples=NUM_SAMPLES)
            # loss
            loss, recA, recB = elbo(q, pA, pB, annealing_factor, lamb=args.lambda_text, beta1=BETA1, beta2=BETA2,
                                    bias=BIAS_TRAIN)
        else:
            shuffled_idx = list(range(args.batch_size))
            random.shuffle(shuffled_idx)
            labels_onehot = labels_onehot[shuffled_idx]
            q = encA(images, num_samples=NUM_SAMPLES)
            q = encB(labels_onehot, num_samples=NUM_SAMPLES, q=q)
            pA = decA(images, {'sharedA': q['sharedA']}, q=q,
                      num_samples=NUM_SAMPLES)
            pB = decB(labels_onehot, {'sharedB': q['sharedB']}, q=q,
                      num_samples=NUM_SAMPLES)
            # epoch_correct += pB['labels_acc_sharedA'].loss.sum().item()
            loss, recA, recB = elbo(q, pA, pB, annealing_factor, lamb=args.lambda_text, beta1=BETA1, beta2=BETA2,
                                    bias=BIAS_TRAIN)

        loss.backward()
        optimizer.step()
        if CUDA:
            loss = loss.cpu()
            recA[0] = recA[0].cpu()
            recB[0] = recB[0].cpu()

        epoch_elbo += loss.item()
        epoch_recA += recA[0].item()
        epoch_recB += recB[0].item()

        if recA[1] is not None:
            if CUDA:
                for i in range(2):
                    recA[i] = recA[i].cpu()
                    recB[i] = recB[i].cpu()
            epoch_rec_poeA += recA[1].item()
            epoch_rec_poeB += recB[1].item()
            pair_cnt += 1

        if b % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)], annealing_factor: {:.3f})'.format(
                e, b * args.batch_size, len(data.dataset),
                   100. * b * args.batch_size / len(data.dataset), annealing_factor))

    if pair_cnt == 0:
        pair_cnt = 1

    print('frac:', cnt / N)

    return epoch_elbo / N, [epoch_recA / N, epoch_rec_poeA / pair_cnt, epoch_rec_crA / pair_cnt], [epoch_recB / N,
                                                                                                   epoch_rec_poeB / pair_cnt,
                                                                                                   epoch_rec_crB / pair_cnt], 1 + epoch_correct / (
               N * args.batch_size)


def test(data, encA, decA, encB, decB, epoch):
    encA.eval()
    decA.eval()
    encB.eval()
    decB.eval()
    epoch_elbo = 0.0
    epoch_correct = 0
    N = 0
    for b, (images, labels) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += 1
            # images = images.view(-1, NUM_PIXELS)
            labels_onehot = torch.zeros(args.batch_size, args.n_shared)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS)
            if CUDA:
                images = images.cuda()
                labels_onehot = labels_onehot.cuda()
            # encode
            q = encA(images, num_samples=NUM_SAMPLES)
            q = encB(labels_onehot, num_samples=NUM_SAMPLES, q=q)
            pA = decA(images, {'sharedA': q['sharedA'], 'sharedB': q['sharedB']}, q=q,
                      num_samples=NUM_SAMPLES)
            pB = decB(labels_onehot, {'sharedB': q['sharedB'], 'sharedA': q['sharedA']}, q=q,
                      num_samples=NUM_SAMPLES, train=False)

            batch_elbo, _, _ = elbo(q, pA, pB, 1, lamb=args.lambda_text, beta1=BETA1, beta2=BETA2, bias=BIAS_TEST)

            if CUDA:
                batch_elbo = batch_elbo.cpu()
            epoch_elbo += batch_elbo.item()
            epoch_correct += pB['labels_sharedA'].loss.sum().item()

    if (epoch + 1) % 20 == 0 or epoch + 1 == args.epochs:
        # util.evaluation.save_traverse(epoch, test_data, encA, decA, CUDA,
        #                               fixed_idxs=[3, 2, 1, 32, 4, 23, 21, 36, 61, 99], output_dir_trvsl=MODEL_NAME,
        #                               flatten_pixel=NUM_PIXELS)

        save_ckpt(e + 1)
    return epoch_elbo / N, 1 + epoch_correct / (N * args.batch_size)


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


def conf_mat(data, encA, decA, encB, decB, epoch):
    encA.eval()
    decA.eval()
    encB.eval()
    decB.eval()

    N = 0
    all_target = []
    all_pred = []

    for b, (images, labels) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += 1
            # images = images.view(-1, NUM_PIXELS)
            labels_onehot = torch.zeros(args.batch_size, args.n_shared)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS)
            if CUDA:
                images = images.cuda()
                labels_onehot = labels_onehot.cuda()
            # encode
            q = encA(images, num_samples=NUM_SAMPLES)
            q = encB(labels_onehot, num_samples=NUM_SAMPLES, q=q)
            _, pred = decB(labels_onehot, {'sharedB': q['sharedB'], 'sharedA': q['sharedA']}, q=q,
                           num_samples=NUM_SAMPLES, train=False)

            if N == 1:
                all_pred = pred.squeeze(0).detach().numpy()
                all_target = labels.detach().numpy()
            else:
                all_pred = np.concatenate((all_pred, pred.squeeze(0).detach().numpy()), axis=0)
                all_target = np.concatenate((all_target, labels.detach().numpy()), axis=0)

    mat = np.zeros((10, 10))
    for i in range(all_target.shape[0]):
        mat[[all_target[i]], all_pred[i]] += 1
    mat = mat.astype(int)

    import matplotlib.pyplot as plt
    import seaborn as sns;
    LABEL_IX_TO_STRING = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress',
                          4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag',
                          9: 'Ankle boot'}

    sns.set(font_scale=1)
    plt.figure(figsize=(6, 6))
    plt.title('FMNIST confusion matrix')
    ax = sns.heatmap(mat, annot=True, fmt="d", xticklabels=list(LABEL_IX_TO_STRING.values()),
                     yticklabels=list(LABEL_IX_TO_STRING.values()))
    print((all_target == all_pred).mean())
    plt.show()

    return (all_target == all_pred).mean()


label_mask = {}

paired_idx = list(range(train_data_size))
random.seed(args.seed)
random.shuffle(paired_idx)
paired_idx = paired_idx[:int(args.label_frac * train_data_size)]
print('paired_idx: ', paired_idx)

for b in range(train_data_size):
    if b in paired_idx:
        label_mask[b] = True
    else:
        label_mask[b] = False

for e in range(args.ckpt_epochs, args.epochs):
    train_start = time.time()
    train_elbo, rec_lossA, rec_lossB, tr_acc = train(train_data, encA, decA, encB, decB,
                                                     optimizer, label_mask, e)
    train_end = time.time()
    test_start = time.time()
    test_elbo, test_accuracy = test(test_data, encA, decA, encB, decB, e)

    if args.viz_on:
        LINE_GATHER.insert(epoch=e,
                           tr_acc=tr_acc,
                           test_acc=test_accuracy,
                           test_total_loss=test_elbo,
                           total_loss=train_elbo,
                           recon_A=rec_lossA[0],
                           recon_poeA=rec_lossA[1],
                           recon_B=rec_lossB[0],
                           recon_poeB=rec_lossB[1],
                           )
        visualize_line()
        LINE_GATHER.flush()

    test_end = time.time()
    print('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e, Accuracy %0.3f (%ds)' % (
        e, train_elbo, train_end - train_start,
        test_elbo, test_accuracy, test_end - test_start))

if args.ckpt_epochs == args.epochs:
    test_accuracy = conf_mat(test_data, encA, decA, encB, decB, args.ckpt_epochs)
    print(test_accuracy)
    # util.evaluation.save_traverse(args.epochs, test_data, encA, decA, CUDA, MODEL_NAME,
    #                               fixed_idxs=[28, 2, 47, 32, 4, 23, 21, 36, 84, 20],
    #                               flatten_pixel=NUM_PIXELS)
    #
    # util.evaluation.save_cross_mnist(args.ckpt_epochs, test_data, encA, decA, encB, 16,
    #                                  args.n_shared, CUDA, MODEL_NAME, flatten_pixel=NUM_PIXELS)

else:
    save_ckpt(args.epochs)
