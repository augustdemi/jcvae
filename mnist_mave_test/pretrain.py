from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import random
import torch
import os
import visdom
import numpy as np

from model import EncoderA, EncoderB, DecoderA, DecoderB
from sklearn.metrics import f1_score

import sys

sys.path.append('../')
import probtorch
import util
from torch.nn import functional as F

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
MODEL_NAME = 'mnist_mvae_pretrain-run_id%d-shared%02d-bs%s-lr%s' % (
    args.run_id, args.n_shared, args.batch_size, args.lr)
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
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['acc'])


def visualize_line():
    data = LINE_GATHER.data
    epoch = torch.Tensor(data['epoch'])
    test_acc = torch.Tensor(data['test_acc'])
    acc = torch.tensor(np.stack([test_acc], -1))

    VIZ.line(
        X=epoch, Y=acc, env=MODEL_NAME + '/lines',
        win=WIN_ID['acc'], update='append',
        opts=dict(xlabel='epoch', ylabel='accuracy',
                  title='Accuracy', legend=['test_acc'])
    )


if args.viz_on:
    WIN_ID = dict(
        acc='win_acc'
    )
    LINE_GATHER = probtorch.util.DataGather(
        'test_acc'
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
if CUDA:
    encA.cuda()
    cuda_tensors(encA)

optimizer = torch.optim.Adam(
    list(encA.parameters()),
    lr=args.lr)


def train(data, encA, epoch, optimizer, fixed_imgs=None, fixed_labels=None):
    encA.train()

    N = 0

    for b, (images, labels) in enumerate(data):
        N += 1
        labels_onehot = torch.zeros(args.batch_size, 10)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS)
        optimizer.zero_grad()
        if CUDA:
            images = images.cuda()
            labels = labels.cuda()
            labels_onehot = labels_onehot.cuda()

        # encode
        q = encA(images, CUDA)
        pred_labels = q['sharedA'].dist.loc.squeeze(0)
        pred_labels = F.log_softmax(pred_labels + EPS, dim=1)
        loss = -(labels_onehot * pred_labels).sum()

        loss.backward()
        optimizer.step()


def test(data, encA):
    encA.eval()
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
            pred_labels = q['sharedA'].dist.loc.squeeze(0)
            pred_labels = F.log_softmax(pred_labels + EPS, dim=1)
            pred_labels = torch.argmax(pred_labels, dim=1)

            if CUDA:
                batch_elbo = batch_elbo.cpu()
                pred_labels = pred_labels.cpu()
                labels = labels.cpu()

            pred_labels = pred_labels.detach().numpy()
            target = labels.detach().numpy()
            epoch_acc += (pred_labels == target).mean()

    return epoch_acc / N


def save_ckpt(e):
    if not os.path.isdir(args.ckpt_path):
        os.mkdir(args.ckpt_path)
    torch.save(encA.state_dict(),
               '%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))


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
    else:
        encA.load_state_dict(torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
                                        map_location=torch.device('cpu')))

mask = {}
fixed_imgs = None
fixed_labels = None
if args.label_frac > 1:
    fixed_imgs, fixed_labels = get_paired_data(args.label_frac, args.seed)

for e in range(args.ckpt_epochs, args.epochs):
    train_start = time.time()
    train(train_data, encA, e, optimizer, fixed_imgs=fixed_imgs, fixed_labels=fixed_labels)
    train_end = time.time()

    test_start = time.time()
    test_accuracy = test(test_data, encA)
    test_end = time.time()

    if args.viz_on:
        LINE_GATHER.insert(epoch=e,
                           test_acc=test_accuracy
                           )
        visualize_line()
        LINE_GATHER.flush()

    if (e + 1) % 20 == 0 or e + 1 == args.epochs:
        save_ckpt(e + 1)

    print('ACC:', test_accuracy)

    # print(
    #     '[Epoch %d] Train: ELBO %.4e (%ds), Test: ELBO %.4e, Accuracy %0.3f (%ds)' % (
    #         e, train_end - train_start,
    #         test_accuracy, test_end - test_start))

save_ckpt(args.epochs)
