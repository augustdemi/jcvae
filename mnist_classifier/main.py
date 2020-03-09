from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import random
import torch
import os
import visdom
import numpy as np
from datasets import DIGIT
from model import EncoderA
from torch.nn import functional as F

import sys

sys.path.append('../')
import probtorch
import util

# ------------------------------------------------
# training parameters

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=37, metavar='N',
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
    parser.add_argument('--epochs', type=int, default=440, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default: 1e-3]')

    parser.add_argument('--label_frac', type=float, default=100.,
                        help='how many labels to use')
    parser.add_argument('--sup_frac', type=float, default=0.4,
                        help='supervision ratio')
    parser.add_argument('--lambda_text', type=float, default=2000.,
                        help='multipler for text reconstruction [default: 10]')
    parser.add_argument('--beta1', type=float, default=3.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--beta2', type=float, default=1.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed for get_paired_data')
    parser.add_argument('--wseed', type=int, default=0, metavar='N',
                        help='random seed for weight')

    parser.add_argument('--ckpt_path', type=str, default='../weights/mnist/0.4/',
                        help='save and load path for ckpt')

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
MODEL_NAME = 'mnist_clf-run_id%d-priv%02ddim-label_frac%s-sup_frac%s-lamb_text%s-beta1%s-beta2%s-seed%s-bs%s-wseed%s' % (
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
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['total_losses'])


def visualize_line():
    data = LINE_GATHER.data
    recon_A = torch.Tensor(data['recon_A'])
    recon_B = torch.Tensor(data['recon_B'])

    recon_poeA = torch.Tensor(data['recon_poeA'])
    recon_poeB = torch.Tensor(data['recon_poeB'])
    recon_crA = torch.Tensor(data['recon_crA'])
    recon_crB = torch.Tensor(data['recon_crB'])
    total_loss = torch.Tensor(data['total_loss'])

    epoch = torch.Tensor(data['epoch'])
    test_acc = torch.Tensor(data['test_acc'])
    test_total_loss = torch.Tensor(data['test_total_loss'])

    llA = torch.tensor(np.stack([recon_A, recon_poeA, recon_crA], -1))
    llB = torch.tensor(np.stack([recon_B, recon_poeB, recon_crB], -1))
    total_losses = torch.tensor(np.stack([total_loss, test_total_loss], -1))

    VIZ.line(
        X=epoch, Y=llA, env=MODEL_NAME + '/lines',
        win=WIN_ID['llA'], update='append',
        opts=dict(xlabel='epoch', ylabel='loglike',
                  title='LL of modalA', legend=['A', 'poeA', 'crA'])
    )
    VIZ.line(
        X=epoch, Y=llB, env=MODEL_NAME + '/lines',
        win=WIN_ID['llB'], update='append',
        opts=dict(xlabel='epoch', ylabel='loglike',
                  title='LL of modalB', legend=['B', 'poeB', 'crB'])
    )

    VIZ.line(
        X=epoch, Y=test_acc, env=MODEL_NAME + '/lines',
        win=WIN_ID['test_acc'], update='append',
        opts=dict(xlabel='epoch', ylabel='accuracy',
                  title='Test Accuracy', legend=['acc'])
    )

    VIZ.line(
        X=epoch, Y=total_losses, env=MODEL_NAME + '/lines',
        win=WIN_ID['total_losses'], update='append',
        opts=dict(xlabel='epoch', ylabel='loss',
                  title='Total Loss', legend=['train_loss', 'test_loss'])
    )


if args.viz_on:
    WIN_ID = dict(
        llA='win_llA', llB='win_llB', test_acc='win_test_acc', total_losses='win_total_losses'
    )
    LINE_GATHER = probtorch.util.DataGather(
        'epoch', 'recon_A', 'recon_B', 'recon_poeA', 'recon_poeB', 'recon_crA', 'recon_crB',
        'total_loss', 'test_total_loss', 'test_acc'
    )
    VIZ = visdom.Visdom(port=args.viz_port)
    viz_init()

train_data = torch.utils.data.DataLoader(DIGIT('./data', train=True), batch_size=args.batch_size, shuffle=True)
test_data = torch.utils.data.DataLoader(DIGIT('./data', train=False), batch_size=args.batch_size, shuffle=False)


def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())


encA = EncoderA(args.wseed)
if CUDA:
    encA.cuda()
    cuda_tensors(encA)

optimizer = torch.optim.Adam(
    list(encA.parameters()),
    lr=args.lr)


def train(data, encA, optimizer):
    encA.train()
    N = 0
    total_loss = 0
    for b, (images, labels) in enumerate(data):
        N += 1
        optimizer.zero_grad()

        labels_onehot = torch.zeros(args.batch_size, 10)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS)
        if CUDA:
            images = images.cuda()
            labels_onehot = labels_onehot.cuda()
        images = images.view(-1, NUM_PIXELS)

        # encode
        pred_attr = encA(images, num_samples=NUM_SAMPLES)

        loss = F.binary_cross_entropy_with_logits(pred_attr, labels_onehot, reduction='none').sum()

        if CUDA:
            loss = loss.cuda()
        loss.backward()
        optimizer.step()
        if CUDA:
            loss = loss.cpu()

        total_loss += loss

        if b % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                e, b * args.batch_size, len(data.dataset),
                   100. * b * args.batch_size / len(data.dataset)))
    return total_loss


def test(data, encA):
    encA.eval()
    epoch_elbo = 0.0
    epoch_acc = 0
    epoch_f1 = 0
    N = 0
    all_pred = []
    all_target = []
    for b, (images, labels) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += 1

            if CUDA:
                images = images.cuda()
            images = images.view(-1, NUM_PIXELS)

            # encode
            pred_attr = encA(images, num_samples=NUM_SAMPLES)

            if CUDA:
                pred_attr = pred_attr.cpu()

            pred = pred_attr.detach().numpy()
            pred = np.argmax(pred, axis=1)
            target = labels.detach().numpy()
            epoch_acc += (pred == target).mean()

            if N == 1:
                all_pred = pred
                all_target = target
            else:
                all_pred = np.concatenate((all_pred, pred), axis=0)
                all_target = np.concatenate((all_target, target), axis=0)

    print('---------------------acc------------------------')
    acc = (all_target == all_pred).mean()
    print(acc)
    print(epoch_acc / N)
    print('-----------------------------------------------')

    return epoch_acc / N, epoch_f1 / N


def save_ckpt(e):
    if not os.path.isdir(args.ckpt_path):
        os.mkdir(args.ckpt_path)
    torch.save(encA.state_dict(),
               '%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))


if args.ckpt_epochs > 0:
    if CUDA:
        encA.load_state_dict(torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs)))
    else:
        encA.load_state_dict(torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
                                        map_location=torch.device('cpu')))

mask = {}
fixed_imgs = None
fixed_attr = None

for e in range(args.ckpt_epochs, args.epochs):
    train_start = time.time()
    total_loss = train(train_data, encA, optimizer)
    train_end = time.time()

    test_start = time.time()
    test_accuracy, test_f1 = test(test_data, encA)
    test_end = time.time()

    if args.viz_on:
        LINE_GATHER.insert(epoch=e,
                           test_f1=test_f1,
                           test_acc=test_accuracy,
                           total_loss=total_loss
                           )
        visualize_line()
        LINE_GATHER.flush()
    if (e + 1) % 10 == 0 or e + 1 == args.epochs:
        save_ckpt(e + 1)

    print(
        '[Epoch %d] Train: ELBO %.4e (%ds), Test: Accuracy %0.3f, F1-score %0.3f (%ds)' % (
            e, total_loss, train_end - train_start, test_accuracy, test_f1, test_end - test_start))

if args.ckpt_epochs == args.epochs:
    # test_elbo, test_accuracy = test(test_data, encA, decA, encB, decB, 0)
    # fixed_idxs=[28, 2, 35, 32, 4, 23, 21, 36, 84, 20], output_dir_trvsl=MODEL_NAME,
    # util.evaluation.mutual_info(test_data, encA, CUDA, flatten_pixel=NUM_PIXELS)

    util.evaluation.mnist_latent(test_data, encA, 1000)


    # ### for stat
    # n_batch = 10
    # random.seed = 0
    # fixed_idxs = random.sample(range(len(train_data.dataset)), 100 * n_batch)
    # fixed_XA = [0] * 100 * n_batch
    # for i, idx in enumerate(fixed_idxs):
    #     fixed_XA[i], _ = train_data.dataset.__getitem__(idx)[:2]
    #     fixed_XA[i] = fixed_XA[i].view(-1, 784)
    #     fixed_XA[i] = fixed_XA[i].squeeze(0)
    #
    # fixed_XA = torch.stack(fixed_XA, dim=0)
    #
    # zS_mean = 0
    # # zS_ori_sum = np.zeros(zS_dim)
    # for idx in range(n_batch):
    #     q = encA(fixed_XA[100 * idx:100 * (idx + 1)], num_samples=1)
    #     zS_mean += q['privateA'].dist.loc
    #     # zS_std += q['sharedA'].dist.scale
    # zS_mean = zS_mean.squeeze(0)
    # min = zS_mean.min(dim=0)[0].detach().cpu().numpy()
    # max = zS_mean.max(dim=0)[0].detach().cpu().numpy()
    # ##############
    #
    # util.evaluation.save_traverse(args.epochs, test_data, encA, decA, CUDA, MODEL_NAME,
    #                               fixed_idxs=[28, 2, 47, 32, 4, 23, 21, 36, 84, 20],
    #                               flatten_pixel=NUM_PIXELS)
    #


    # util.evaluation.save_cross_mnist(args.ckpt_epochs, test_data, encA, decA, encB, 16,
    #                                  args.n_shared, CUDA, MODEL_NAME, flatten_pixel=NUM_PIXELS)

    # util.evaluation.save_reconst(args.epochs, test_data, encA, decA, encB, decB, CUDA, fixed_idxs=[3, 2, 1, 30, 4, 23, 21, 41, 84, 99], output_dir_trvsl=MODEL_NAME, flatten_pixel=NUM_PIXELS)

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
