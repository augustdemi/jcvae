from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import random
import torch
import os
import visdom
import numpy as np

from model import EncoderA, EncoderB, DecoderA, DecoderB
from datasets import datasets
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
    parser.add_argument('--run_id', type=int, default=2, metavar='N',
                        help='run_id')
    parser.add_argument('--run_desc', type=str, default='',
                        help='run_id desc')
    parser.add_argument('--n_shared', type=int, default=100,
                        help='size of the latent embedding of shared')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--ckpt_epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate [default: 1e-3]')

    parser.add_argument('--label_frac', type=float, default=1.,
                        help='how many labels to use')
    parser.add_argument('--sup_frac', type=float, default=1.,
                        help='supervision ratio')
    parser.add_argument('--lambda_text', type=float, default=10.,
                        help='multipler for text reconstruction [default: 10]')
    parser.add_argument('--beta1', type=float, default=1.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--beta2', type=float, default=1.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed for get_paired_data')
    parser.add_argument('--wseed', type=int, default=0, metavar='N',
                        help='random seed for weight')

    parser.add_argument('--ckpt_path', type=str, default='../weights/celeba_mvae/1',
                        help='save and load path for ckpt')
    parser.add_argument('--annealing-epochs', type=int, default=20, metavar='N',
                        help='number of epochs to anneal KL for [default: 20]')
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
MODEL_NAME = 'celeba_mvae-run_id%d-shared%02ddim-label_frac%s-sup_frac%s-lamb_text%s-beta1%s-beta2%s-seed%s-bs%s-wseed%s-lr%s' % (
    args.run_id, args.n_shared, args.label_frac, args.sup_frac, args.lambda_text, args.beta1,
    args.beta2, args.seed,
    args.batch_size, args.wseed, args.lr)
DATA_PATH = '../data'
ATTR_TO_PLOT = ['Heavy_Makeup', 'Male', 'Mouth_Slightly_Open', 'Smiling', 'Straight_Hair', 'Eyeglasses', 'Bangs', 'off']

ATTR_TO_IX_DICT = {'Sideburns': 30, 'Black_Hair': 8, 'Wavy_Hair': 33, 'Young': 39, 'Heavy_Makeup': 18,
                   'Blond_Hair': 9, 'Attractive': 2, '5_o_Clock_Shadow': 0, 'Wearing_Necktie': 38,
                   'Blurry': 10, 'Double_Chin': 14, 'Brown_Hair': 11, 'Mouth_Slightly_Open': 21,
                   'Goatee': 16, 'Bald': 4, 'Pointy_Nose': 27, 'Gray_Hair': 17, 'Pale_Skin': 26,
                   'Arched_Eyebrows': 1, 'Wearing_Hat': 35, 'Receding_Hairline': 28, 'Straight_Hair': 32,
                   'Big_Nose': 7, 'Rosy_Cheeks': 29, 'Oval_Face': 25, 'Bangs': 5, 'Male': 20, 'Mustache': 22,
                   'High_Cheekbones': 19, 'No_Beard': 24, 'Eyeglasses': 15, 'Bags_Under_Eyes': 3,
                   'Wearing_Necklace': 37, 'Wearing_Lipstick': 36, 'Big_Lips': 6, 'Narrow_Eyes': 23,
                   'Chubby': 13, 'Smiling': 31, 'Bushy_Eyebrows': 12, 'Wearing_Earrings': 34}
# we only keep 18 of the more visually distinctive features
# See [1] Perarnau, Guim, et al. "Invertible conditional gans for
#         image editing." arXiv preprint arXiv:1611.06355 (2016).
ATTR_IX_TO_KEEP = [4, 5, 8, 9, 11, 12, 15, 17, 18, 20, 21, 22, 26, 28, 31, 32, 33, 35]
IX_TO_ATTR_DICT = {v: k for k, v in ATTR_TO_IX_DICT.items()}


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
N_ATTR = 18


# visdom setup
def viz_init():
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llA'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['llB'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['acc'])
    VIZ.close(env=MODEL_NAME + '/lines', win=WIN_ID['f1'])
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
    val_acc = torch.Tensor(data['val_acc'])
    test_total_loss = torch.Tensor(data['test_total_loss'])
    val_total_loss = torch.Tensor(data['val_total_loss'])
    test_f1 = torch.Tensor(data['test_f1'])
    val_f1 = torch.Tensor(data['val_f1'])

    llA = torch.tensor(np.stack([recon_A, recon_poeA], -1))
    llB = torch.tensor(np.stack([recon_B, recon_poeB], -1))
    total_losses = torch.tensor(np.stack([total_loss, test_total_loss, val_total_loss], -1))
    acc = torch.tensor(np.stack([test_acc, val_acc], -1))
    f1 = torch.tensor(np.stack([test_f1, val_f1], -1))

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
                  title='Accuracy', legend=['test_acc', 'val_acc'])
    )

    VIZ.line(
        X=epoch, Y=f1, env=MODEL_NAME + '/lines',
        win=WIN_ID['f1'], update='append',
        opts=dict(xlabel='epoch', ylabel='accuracy',
                  title='F1 score', legend=['test_f1', 'val_f1'])
    )

    VIZ.line(
        X=epoch, Y=total_losses, env=MODEL_NAME + '/lines',
        win=WIN_ID['total_losses'], update='append',
        opts=dict(xlabel='epoch', ylabel='loss',
                  title='Total Loss', legend=['train_loss', 'test_loss', 'val_loss'])
    )


if args.viz_on:
    WIN_ID = dict(
        llA='win_llA', llB='win_llB', acc='win_acc', total_losses='win_total_losses', f1='win_f1'
    )
    LINE_GATHER = probtorch.util.DataGather(
        'epoch', 'recon_A', 'recon_B', 'recon_poeA', 'recon_poeB',
        'total_loss', 'test_total_loss', 'test_acc', 'val_total_loss', 'val_acc', 'test_f1', 'val_f1'
    )
    VIZ = visdom.Visdom(port=args.viz_port)
    viz_init()

preprocess_data = transforms.Compose([
    transforms.CenterCrop((168, 178)),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_data = torch.utils.data.DataLoader(datasets(partition='train', data_dir='../../data/celeba2',
                                                  image_transform=preprocess_data), batch_size=args.batch_size,
                                         shuffle=True)

test_data = torch.utils.data.DataLoader(datasets(partition='test', data_dir='../../data/celeba2',
                                                 image_transform=preprocess_data), batch_size=args.batch_size,
                                        shuffle=False)
val_data = torch.utils.data.DataLoader(datasets(partition='val', data_dir='../../data/celeba2',
                                                image_transform=preprocess_data), batch_size=args.batch_size,
                                       shuffle=False)

print('>>> data loaded')
print('train: ', len(train_data.dataset))
print('val: ', len(val_data.dataset))
print('test: ', len(test_data.dataset))

BIAS_TRAIN = (len(train_data.dataset) - 1) / (args.batch_size - 1)
BIAS_VAL = (len(val_data.dataset) - 1) / (args.batch_size - 1)
BIAS_TEST = (len(test_data.dataset) - 1) / (args.batch_size - 1)


def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())


encA = EncoderA(args.wseed, zShared_dim=args.n_shared)
decA = DecoderA(args.wseed, zShared_dim=args.n_shared)
encB = EncoderB(args.wseed, num_attr=N_ATTR, zShared_dim=args.n_shared)
decB = DecoderB(args.wseed, num_attr=N_ATTR, zShared_dim=args.n_shared)
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


def elbo(q, pA, pB, lamb=1.0, annealing_factor=1.0):

    muA_own = q['sharedA'].dist.loc.squeeze(0)
    stdA_own = q['sharedA'].dist.scale.squeeze(0)
    muB_own = q['sharedB'].dist.loc.squeeze(0)
    stdB_own = q['sharedB'].dist.scale.squeeze(0)

    # from each of modality
    reconst_loss_A = pA['images_own'].loss.mean()
    kl_A = -0.5 * torch.sum(1 + torch.log(stdA_own ** 2 + EPS) - muA_own.pow(2) - torch.log(stdA_own ** 2 + EPS).exp(),
                            dim=1)
    kl_A = kl_A.mean()
    reconst_loss_B = pB['attr_own'].loss.mean()
    kl_B = -0.5 * torch.sum(1 + torch.log(stdB_own ** 2 + EPS) - muB_own.pow(2) - torch.log(stdB_own ** 2 + EPS).exp(),
                            dim=1)
    kl_B = kl_B.mean()

    if q['poe'] is not None:
        mu_poe = q['poe'].dist.loc.squeeze(0)
        stdB_poe = q['poe'].dist.scale.squeeze(0)
        reconst_loss_poeA = pA['images_poe'].loss.mean()
        reconst_loss_poeB = pB['attr_poe'].loss.mean()
        kl_poe = -0.5 * torch.sum(
            1 + torch.log(stdB_poe ** 2 + EPS) - mu_poe.pow(2) - torch.log(stdB_poe ** 2 + EPS).exp(), dim=1)
        kl_poe = kl_poe.mean()
        loss = (reconst_loss_A - annealing_factor * kl_A) + (lamb * reconst_loss_B - annealing_factor * kl_B) + (
            reconst_loss_poeA + lamb * reconst_loss_poeB - annealing_factor * kl_poe)
    else:
        reconst_loss_poeA = reconst_loss_poeB = None
        loss = 2 * ((reconst_loss_A - kl_A) + (lamb * reconst_loss_B - annealing_factor * kl_B))

    return -loss, [reconst_loss_A, reconst_loss_poeA], [reconst_loss_B, reconst_loss_poeB]


def train(data, encA, decA, encB, decB, epoch, optimizer,
          label_mask={}, fixed_imgs=None, fixed_attr=None):
    epoch_elbo = 0.0
    epoch_recA = epoch_rec_poeA = epoch_rec_crA = 0.0
    epoch_recB = epoch_rec_poeB = epoch_rec_crB = 0.0
    pair_cnt = 0
    encA.train()
    encB.train()
    decA.train()
    decB.train()
    N = 0
    torch.autograd.set_detect_anomaly(True)
    for b, (images, attributes) in enumerate(data):
        if epoch < args.annealing_epochs:
            # compute the KL annealing factor for the current mini-batch in the current epoch
            annealing_factor = (float(b + epoch * len(train_data) + 1) /
                                float(args.annealing_epochs * len(train_data)))
        else:
            # by default the KL annealing factor is unity
            annealing_factor = 1.0

        if args.label_frac > 1 and random.random() < args.sup_frac:
            # print(b)
            N += 1
            shuffled_idx = list(range(int(args.label_frac)))
            random.shuffle(shuffled_idx)
            shuffled_idx = shuffled_idx[:args.batch_size]
            # print(shuffled_idx[:10])
            images = fixed_imgs[shuffled_idx]
            attributes = fixed_attr[shuffled_idx]
            optimizer.zero_grad()
            if CUDA:
                images = images.cuda()
                attributes = attributes.cuda()

            # encode
            q = encA(images, CUDA, num_samples=NUM_SAMPLES)
            q = encB(attributes, CUDA, num_samples=NUM_SAMPLES, q=q)

            ## poe ##
            mu_poe, std_poe = probtorch.util.apply_poe(CUDA, q['sharedA'].dist.loc, q['sharedA'].dist.scale,
                                                       q['sharedB'].dist.loc, q['sharedB'].dist.scale)
            q.normal(mu_poe,
                     std_poe,
                     name='poe')

            muA, stdA = probtorch.util.apply_poe(CUDA, q['sharedA'].dist.loc, q['sharedA'].dist.scale)
            q['sharedA'].dist.loc = muA
            q['sharedA'].dist.scale = stdA

            muB, stdB = probtorch.util.apply_poe(CUDA, q['sharedB'].dist.loc, q['sharedB'].dist.scale)
            q['sharedB'].dist.loc = muB
            q['sharedB'].dist.scale = stdB

            # decode attr
            shared_dist = {'poe': 'poe', 'own': 'sharedB'}
            pB, _ = decB(attributes, shared_dist, q=q, num_samples=NUM_SAMPLES)

            # decode img
            shared_dist = {'poe': 'poe', 'own': 'sharedA'}
            pA = decA(images, shared_dist, q=q, num_samples=NUM_SAMPLES)

            for param in encB.parameters():
                param.requires_grad = True
            for param in decB.parameters():
                param.requires_grad = True
            # loss
            loss, recA, recB = elbo(q, pA, pB, annealing_factor=annealing_factor)
        else:
            N += 1
            if CUDA:
                images = images.cuda()
                attributes = attributes.cuda()
            optimizer.zero_grad()

            if b not in label_mask:
                label_mask[b] = (random.random() < args.label_frac)

            if (label_mask[b] and args.label_frac == args.sup_frac):
                # encode
                q = encA(images, CUDA, num_samples=NUM_SAMPLES)
                q = encB(attributes, CUDA, num_samples=NUM_SAMPLES, q=q)

                ## poe ##
                mu_poe, std_poe = probtorch.util.apply_poe(CUDA, q['sharedA'].dist.loc, q['sharedA'].dist.scale,
                                                           q['sharedB'].dist.loc, q['sharedB'].dist.scale)
                q.normal(mu_poe,
                         std_poe,
                         name='poe')

                muA, stdA = probtorch.util.apply_poe(CUDA, q['sharedA'].dist.loc, q['sharedA'].dist.scale)
                q['sharedA'].dist.loc = muA
                q['sharedA'].dist.scale = stdA

                muB, stdB = probtorch.util.apply_poe(CUDA, q['sharedB'].dist.loc, q['sharedB'].dist.scale)
                q['sharedB'].dist.loc = muB
                q['sharedB'].dist.scale = stdB


                # decode attr
                shared_dist = {'poe': 'poe', 'own': 'sharedB'}
                pB, _ = decB(attributes, shared_dist, q=q, num_samples=NUM_SAMPLES)

                # decode img
                shared_dist = {'poe': 'poe', 'own': 'sharedA'}
                pA = decA(images, shared_dist, q=q, num_samples=NUM_SAMPLES)

                for param in encB.parameters():
                    param.requires_grad = True
                for param in decB.parameters():
                    param.requires_grad = True
                # loss
                loss, recA, recB = elbo(q, pA, pB, lamb=args.lambda_text, annealing_factor=annealing_factor)
            else:
                shuffled_idx = list(range(args.batch_size))
                random.shuffle(shuffled_idx)
                attributes = attributes[shuffled_idx]

                # encode
                q = encA(images, CUDA, num_samples=NUM_SAMPLES)
                q = encB(attributes, CUDA, num_samples=NUM_SAMPLES, q=q)

                muA, stdA = probtorch.util.apply_poe(CUDA, q['sharedA'].dist.loc, q['sharedA'].dist.scale)
                q['sharedA'].dist.loc = muA
                q['sharedA'].dist.scale = stdA

                muB, stdB = probtorch.util.apply_poe(CUDA, q['sharedB'].dist.loc, q['sharedB'].dist.scale)
                q['sharedB'].dist.loc = muB
                q['sharedB'].dist.scale = stdB

                # decode attr
                shared_dist = {'own': 'sharedB'}
                pB, _ = decB(attributes, shared_dist, q=q, num_samples=NUM_SAMPLES)

                # decode img
                shared_dist = {'own': 'sharedA'}
                pA = decA(images, shared_dist, q=q, num_samples=NUM_SAMPLES)
                for param in encB.parameters():
                    param.requires_grad = False
                for param in decB.parameters():
                    param.requires_grad = False
                loss, recA, recB = elbo(q, pA, pB, lamb=args.lambda_text, annealing_factor=annealing_factor)

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
    return epoch_elbo / N, [epoch_recA / N, epoch_rec_poeA / pair_cnt], [epoch_recB / N,
                                                                         epoch_rec_poeB / pair_cnt], label_mask


def test(data, encA, decA, encB, decB, epoch, bias):
    encA.eval()
    decA.eval()
    encB.eval()
    decB.eval()
    epoch_elbo = 0.0
    epoch_acc = 0
    epoch_f1 = 0
    N = 0
    all_pred = []
    all_target = []
    for b, (images, attributes) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += 1
            if CUDA:
                images = images.cuda()
                attributes = attributes.cuda()

            # encode
            q = encA(images, CUDA, num_samples=NUM_SAMPLES)
            q = encB(attributes, CUDA, num_samples=NUM_SAMPLES, q=q)

            # poe for each modal
            muA, stdA = probtorch.util.apply_poe(CUDA, q['sharedA'].dist.loc, q['sharedA'].dist.scale)
            q['sharedA'].dist.loc = muA
            q['sharedA'].dist.scale = stdA

            muB, stdB = probtorch.util.apply_poe(CUDA, q['sharedB'].dist.loc, q['sharedB'].dist.scale)
            q['sharedB'].dist.loc = muB
            q['sharedB'].dist.scale = stdB


            # decode attr
            shared_dist = {'own': 'sharedB', 'cross': 'sharedA'}
            pB, pred_attr = decB(attributes, shared_dist, q=q, num_samples=NUM_SAMPLES, train=False)

            # decode img
            shared_dist = {'own': 'sharedA', 'cross': 'sharedB'}
            pA = decA(images, shared_dist, q=q, num_samples=NUM_SAMPLES)

            batch_elbo, _, _ = elbo(q, pA, pB, lamb=args.lambda_text)

            if CUDA:
                batch_elbo = batch_elbo.cpu()
                pred_attr = pred_attr.cpu()
                attributes = attributes.cpu()
            epoch_elbo += batch_elbo.item()

            pred = pred_attr.detach().numpy()
            pred = np.round(np.exp(pred))
            target = attributes.detach().numpy()
            epoch_acc += (pred == target).mean()
            epoch_f1 += f1_score(target, pred, average="samples")

            if N == 1:
                all_pred = pred
                all_target = target
            else:
                all_pred = np.concatenate((all_pred, pred), axis=0)
                all_target = np.concatenate((all_target, target), axis=0)
    print('---------------------f1------------------------')
    f1 = []
    for i in range(18):
        f1.append(f1_score(all_target[:, i], all_pred[:, i], average="binary"))

    f1 = list(enumerate(f1))
    f1.sort(key=lambda f1: f1[1])

    for i in range(18):
        print(IX_TO_ATTR_DICT[ATTR_IX_TO_KEEP[f1[i][0]]], f1[i][1])
    print('---------------------acc------------------------')

    all_acc = []
    for i in range(18):
        acc = (all_target[:, i] == all_pred[:, i]).mean()
        all_acc.append(acc)

    all_acc = list(enumerate(all_acc))
    all_acc.sort(key=lambda all_acc: all_acc[1])

    for i in range(18):
        print(IX_TO_ATTR_DICT[ATTR_IX_TO_KEEP[all_acc[i][0]]], all_acc[i][1])
    print('-----------------------------------------------')
    return epoch_elbo / N, epoch_acc / N, epoch_f1 / N


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

mask = {}
fixed_imgs = None
fixed_attr = None
if args.label_frac > 1:
    fixed_imgs, fixed_attr = get_paired_data(args.label_frac, args.seed)

for e in range(args.ckpt_epochs, args.epochs):
    train_start = time.time()
    train_elbo, rec_lossA, rec_lossB, mask = train(train_data, encA, decA, encB, decB, e,
                                                   optimizer, mask, fixed_imgs=fixed_imgs, fixed_attr=fixed_attr)
    train_end = time.time()

    val_start = time.time()
    val_elbo, val_accuracy, val_f1 = test(val_data, encA, decA, encB, decB, e, BIAS_VAL)
    val_end = time.time()

    test_start = time.time()
    test_elbo, test_accuracy, test_f1 = test(test_data, encA, decA, encB, decB, e, BIAS_TEST)
    test_end = time.time()

    if args.viz_on:
        LINE_GATHER.insert(epoch=e,
                           test_f1=test_f1,
                           val_f1=val_f1,
                           test_acc=test_accuracy,
                           val_acc=val_accuracy,
                           test_total_loss=test_elbo,
                           val_total_loss=val_elbo,
                           total_loss=train_elbo,
                           recon_A=rec_lossA[0],
                           recon_poeA=rec_lossA[1],
                           recon_B=rec_lossB[0],
                           recon_poeB=rec_lossB[1]
                           )
        visualize_line()
        LINE_GATHER.flush()

    if (e + 1) % 5 == 0 or e + 1 == args.epochs:
        save_ckpt(e + 1)
        encB.eval()
        decA.eval()
        util.evaluation.save_cross_celeba_mvae(e, decA, encB, ATTR_TO_PLOT, 64,
                                               N_ATTR, CUDA, MODEL_NAME)

    print(
        '[Epoch %d] Train: ELBO %.4e (%ds), Val: ELBO %.4e (%ds), Test: ELBO %.4e, Accuracy %0.3f, F1-score %0.3f (%ds)' % (
            e, train_elbo, train_end - train_start, val_elbo, val_end - val_start,
            test_elbo, test_accuracy, test_f1, test_end - test_start))

if args.ckpt_epochs == args.epochs:
    encB.eval()
    decA.eval()
    # util.evaluation.save_cross_celeba_mvae(args.ckpt_epochs, decA, encB, ATTR_TO_PLOT, 64,
    #                                        N_ATTR, CUDA, MODEL_NAME)

    test_elbo, test_accuracy, test_f1 = test(test_data, encA, decA, encB, decB, 0, BIAS_TEST)

    print('test_accuracy:', test_accuracy)
    print('test_f1:', test_f1)

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
