from torchvision import datasets, transforms

import time
import random
import torch
import os
import numpy as np
from model import EncoderA, DecoderA
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
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
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
MODEL_NAME = 'cub-img-ae-run_id%d-bs%d' % (
    args.run_id, args.batch_size)
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
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['total_losses'])


def visualize_line():
    data = LINE_GATHER.data

    epoch = torch.Tensor(data['epoch'])
    test_total_loss = torch.Tensor(data['test_total_loss'])
    total_loss = torch.Tensor(data['total_loss'])

    total_losses = torch.tensor(np.stack([total_loss, test_total_loss], -1))

    VIZ.line(
        X=epoch, Y=total_losses, env=MODEL_NAME + '/lines',
        win=WIN_ID['total_losses'], update='append',
        opts=dict(xlabel='epoch', ylabel='loss',
                  title='Total Loss', legend=['train_loss', 'test_loss'])
    )


if args.viz_on:
    WIN_ID = dict(
        total_losses='win_total_losses'
    )
    LINE_GATHER = probtorch.util.DataGather(
        'epoch',
        'total_loss', 'test_total_loss'
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


encA = EncoderA(args.wseed)
decA = DecoderA(args.wseed)

if CUDA:
    encA.cuda()
    decA.cuda()
    cuda_tensors(encA)
    cuda_tensors(decA)
    if len(args.gpu) > 2:
        print('multi: ' + args.gpu)
        encA = nn.DataParallel(encA)
        decA = nn.DataParallel(decA)

optimizer = torch.optim.Adam(
    list(
        encA.parameters()) + list(decA.parameters()),
    lr=args.lr)


def recon_loss(recon, orig):
    return -(torch.log(recon + EPS) * orig +
             torch.log(1 - recon + EPS) * (1 - orig)).sum(-1)


def train(data, encA, decA, optimizer):
    epoch_loss = 0.0
    encA.train()
    decA.train()

    N = 0

    torch.autograd.set_detect_anomaly(True)
    for b, (images, _, _) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += 1
            optimizer.zero_grad()
            if CUDA:
                images = images.cuda()
            feature = encA(images)
            recon_images = decA(feature)
            recon_images = recon_images.view(recon_images.size(0), -1)
            images = images.view(images.size(0), -1)
            loss = recon_loss(recon_images, images).mean()
            loss.backward()
            optimizer.step()
            if CUDA:
                loss = loss.cpu()
            epoch_loss += loss.item()
    return epoch_loss / N


def test(data, encA, decA):
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
    torch.save(decA, '%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, e))


####
if args.ckpt_epochs > 0:
    if CUDA:
        encA = torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs))
        decA = torch.load('%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs))
    else:
        encA = torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs), map_location='cpu')
        decA = torch.load('%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs), map_location='cpu')

for e in range(args.ckpt_epochs, args.epochs):

    train_start = time.time()
    train_loss = train(train_data, encA, decA, optimizer)
    train_end = time.time()
    test_start = time.time()
    test_loss = test(test_data, encA, decA)
    test_end = time.time()

    if args.viz_on:
        LINE_GATHER.insert(epoch=e,
                           test_total_loss=test_loss,
                           total_loss=train_loss,
                           )
        visualize_line()
        LINE_GATHER.flush()

    if (e + 1) % 10 == 0 or e + 1 == args.epochs:
        save_ckpt(e + 1)
        util.evaluation.save_recon_cub_ae(e, test_data, encA, decA, CUDA, MODEL_NAME,
                                          fixed_idxs=[658, 1570, 2233, 2456, 2880, 1344, 2750, 1800, 1111, 300, 700,
                                                      1270, 2133, 2856, 2680, 1300])
        util.evaluation.save_recon_cub_ae(e, train_data, encA, decA, CUDA, MODEL_NAME,
                                          fixed_idxs=[130, 215, 502, 537, 4288, 1000, 2400, 1220, 3002, 3312, 230, 415,
                                                      602, 737, 3288, 1500])

    print('[Epoch %d] Train loss: %.4e (%ds) Test loss: %.4e (%ds)' % (
        e, train_loss, train_end - train_start,
        test_loss, test_end - test_start))

if args.ckpt_epochs == args.epochs:
    util.evaluation.save_recon_cub_ae(args.epochs, test_data, encA, decA, CUDA, MODEL_NAME,
                                      fixed_idxs=[658, 1570, 2233, 2456, 2880, 1344, 2750, 1800, 1111, 300, 700, 1270,
                                                  2133, 2856, 2680, 1300])
    util.evaluation.save_recon_cub_ae(args.epochs, train_data, encA, decA, CUDA, MODEL_NAME,
                                      fixed_idxs=[130, 215, 502, 537, 4288, 1000, 2400, 1220, 3002, 3312, 230, 415, 602,
                                                  737, 3288, 1500])

else:
    save_ckpt(args.epochs)
