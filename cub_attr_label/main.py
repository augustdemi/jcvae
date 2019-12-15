from torchvision import datasets, transforms

import time
import random
import torch
import os
import numpy as np
from model import linearRegression
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
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
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
MODEL_NAME = 'cub_reg-run_id%d-privA%02ddim-lamb%s-beta%s-lr%s-bs%s-wseed%s-seed%s' % (
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

primary_attr = ['bill_length', 'shape', 'breast_pattern', 'belly_pattern', 'bill_shape',
                'bill_color', 'throat_color', 'crown_color', 'forehead_color', 'underparts_color', 'primary_color',
                'breast_color', 'wing_color', 'belly_color', 'wing_pattern']
print(primary_attr)
ATTR_IDX = []
ATTR_DIM = []
N_ATTR = len(primary_attr)
TRAIN_CLASSES = np.genfromtxt(path + 'attributes/trainvalids.txt', delimiter='\n', dtype=int)

attributes = np.genfromtxt(path + 'attributes/attr.txt', delimiter='\n', dtype=str)
for i in range(attributes.shape[0]):
    if attributes[i].split("::")[0] in primary_attr:
        ATTR_IDX.append(i)
        ATTR_DIM.append(len(attributes[i].split("::")[1].split(',')) + 1)

ATTR_PRIOR = [ATTR_PRIOR[i] for i in ATTR_IDX]


# visdom setup
def viz_init():
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['train_acc'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['test_acc'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['total_losses'])


def visualize_line():
    data = LINE_GATHER.data
    epoch = torch.Tensor(data['epoch'])
    train_acc = torch.Tensor(data['train_acc'])
    test_acc = torch.Tensor(data['test_acc'])
    test_total_loss = torch.Tensor(data['test_total_loss'])
    total_loss = torch.Tensor(data['total_loss'])

    total_losses = torch.tensor(np.stack([total_loss, test_total_loss], -1))

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
        train_acc='win_train_acc', test_acc='win_test_acc',
        total_losses='win_total_losses'
    )
    LINE_GATHER = probtorch.util.DataGather(
        'epoch',
        'total_loss', 'test_total_loss', 'train_acc', 'test_acc'
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


regressor = linearRegression(args.wseed, zSharedAttr_dim=ATTR_DIM, zSharedLabel_dim=N_CLASSES)
if CUDA:
    regressor.cuda()
    cuda_tensors(regressor)

optimizer = torch.optim.Adam(
    list(regressor.parameters()),
    lr=args.lr)


def train(data, regressor, optimizer):
    epoch_elbo = 0.0
    regressor.train()

    N = 0
    epoch_correct = 0

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
                labels_onehot = labels_onehot.cuda()
                attributes = attributes.cuda()
                for i in range(len(attr)):
                    attr[i] = attr[i].cuda()
            # encode

            loss, acc = regressor(attributes, labels_onehot)
            loss.backward()
            optimizer.step()
            if CUDA:
                loss = loss.cpu()
                acc = acc.cpu()
            epoch_elbo += loss.item()
            epoch_correct += acc.item()

    return epoch_elbo / N, epoch_correct / (N * args.batch_size)


def test(data, regressor, epoch):
    epoch_elbo = 0.0
    regressor.eval()

    N = 0
    epoch_correct = 0

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
                labels_onehot = labels_onehot.cuda()
                attributes = attributes.cuda()
                for i in range(len(attr)):
                    attr[i] = attr[i].cuda()
            # encode

            loss, acc = regressor(attributes, labels_onehot)
            if CUDA:
                loss = loss.cpu()
                acc = acc.cpu()
            epoch_elbo += loss.item()
            epoch_correct += acc.item()

    return epoch_elbo / N, epoch_correct / (N * args.batch_size)


####
def save_ckpt(e):
    torch.save(regressor, '%s/%s-regressor_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))


####
if args.ckpt_epochs > 0:
    if CUDA:
        encA = torch.load('%s/%s-regressor_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs))
    else:
        encA = torch.load('%s/%s-regressor_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
                          map_location='cpu')

for e in range(args.ckpt_epochs, args.epochs):
    train_start = time.time()
    train_loss, train_acc = train(train_data, regressor, optimizer)
    train_end = time.time()
    test_start = time.time()
    test_loss, test_acc = train(test_data, regressor, optimizer)

    if args.viz_on:
        LINE_GATHER.insert(epoch=e,
                           train_acc=train_acc,
                           test_acc=test_acc,
                           test_total_loss=test_loss,
                           total_loss=train_loss
                           )
        visualize_line()
        LINE_GATHER.flush()

    test_end = time.time()
    if (e + 1) % 10 == 0 or e + 1 == args.epochs:
        save_ckpt(e + 1)
    print('[Epoch %d] Train: ELBO %.4e (%ds) Test: TR Accuracy %0.3f, TE Accuracy %0.3f (%ds)' % (
        e, train_loss, train_end - train_start,
        train_acc, test_acc, test_end - test_start))

save_ckpt(args.epochs)
