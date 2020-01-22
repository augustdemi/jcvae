from torchvision import datasets, transforms
import os
import torch
import random

from model import Encoder, Decoder

import sys

sys.path.append('../')
import probtorch
import util

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=0, metavar='N',
                        help='run_id')
    parser.add_argument('--run_desc', type=str, default='',
                        help='run_id desc')
    parser.add_argument('--n_shared', type=int, default=10,
                        help='size of the latent embedding of shared')
    parser.add_argument('--n_private', type=int, default=50,
                        help='size of the latent embedding of private')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--ckpt_epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default: 1e-3]')

    parser.add_argument('--label_frac', type=float, default=100,
                        help='how many labels to use')
    parser.add_argument('--sup_frac', type=float, default=0.02,
                        help='supervision ratio')
    parser.add_argument('--lambda_text', type=float, default=-1.,
                        help='multipler for text reconstruction [default: 10]')
    parser.add_argument('--beta', type=float, default=-1.,
                        help='multipler for TC [default: 10]')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed for get_paired_data')

    parser.add_argument('--ckpt_path', type=str, default='../weights/theirs',
                        help='save and load path for ckpt')

    args = parser.parse_args()

# ------------------------------------------------


EPS = 1e-9
CUDA = torch.cuda.is_available()

# path parameters
MODEL_NAME = 'mnist-run_id%d-priv%02ddim-label_frac%s-sup_frac%s-lamb_text%s-beta%s-seed%s' % (
args.run_id, args.n_private, args.label_frac, args.sup_frac, args.lambda_text, args.beta, args.seed)
DATA_PATH = '../data'

if len(args.run_desc) > 1:
    desc_file = os.path.join(args.ckpt_path, 'run_id' + str(args.run_id) + '.txt')
    with open(desc_file, 'w') as outfile:
        outfile.write(args.run_desc)

BETA = (1., args.beta, 1.)
BIAS_TRAIN = (60000 - 1) / (args.batch_size - 1)
BIAS_TEST = (10000 - 1) / (args.batch_size - 1)
# model parameters
NUM_PIXELS = None
TEMP = 0.66
NUM_SAMPLES = 1

if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)

train_data = torch.utils.data.DataLoader(
    datasets.SVHN(DATA_PATH, split='train', download=True,
                  transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True)
test_data = torch.utils.data.DataLoader(
    datasets.SVHN(DATA_PATH, split='test', download=True,
                  transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True)


def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())


enc = Encoder(num_digits=10, num_style=args.n_private)
dec = Decoder(num_digits=10, num_style=args.n_private)
if CUDA:
    enc.cuda()
    dec.cuda()
    cuda_tensors(enc)
    cuda_tensors(dec)

optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                             lr=args.lr)


def elbo(q, p, alpha=0.1):
    if NUM_SAMPLES is None:
        return probtorch.objectives.montecarlo.elbo(q, p, sample_dim=None, batch_dim=0, alpha=alpha)
    else:
        return probtorch.objectives.montecarlo.elbo(q, p, sample_dim=0, batch_dim=1, alpha=alpha)


FIXED = []


def train(data, enc, dec, optimizer,
          label_mask={}, label_fraction=args.label_frac, fixed_imgs=None, fixed_labels=None):
    epoch_elbo = 0.0
    enc.train()
    dec.train()
    N = 0
    for b, (images, labels) in enumerate(data):
        if images.size()[0] == args.batch_size:
            if args.label_frac > 1 and random.random() < args.sup_frac:
                # print(b)
                N += 1
                shuffled_idx = list(range(int(args.label_frac)))
                random.shuffle(shuffled_idx)
                shuffled_idx = shuffled_idx[:args.batch_size]
                # print(shuffled_idx[:10])
                fixed_imgs_batch = fixed_imgs[shuffled_idx]
                fixed_labels_batch = fixed_labels[shuffled_idx]
                labels_onehot = torch.zeros(args.batch_size, args.n_shared)
                labels_onehot.scatter_(1, fixed_labels_batch.unsqueeze(1), 1)
                labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS)

                optimizer.zero_grad()
                if CUDA:
                    images = images.cuda()
                    labels_onehot = labels_onehot.cuda()
                optimizer.zero_grad()
                q = enc(images, labels_onehot, num_samples=NUM_SAMPLES)
                p = dec(images, q, num_samples=NUM_SAMPLES)
                loss = -elbo(q, p)
            else:
                N += args.batch_size
                labels_onehot = torch.zeros(args.batch_size, args.n_shared)
                labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
                labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS)
                if CUDA:
                    images = images.cuda()
                    labels_onehot = labels_onehot.cuda()
                optimizer.zero_grad()
                if b not in label_mask:
                    label_mask[b] = (random.random() < label_fraction)
                if (label_mask[b] and args.label_frac == args.sup_frac):
                    q = enc(images, labels_onehot, num_samples=NUM_SAMPLES)
                else:
                    q = enc(images, num_samples=NUM_SAMPLES)
                p = dec(images, q, num_samples=NUM_SAMPLES)
                loss = -elbo(q, p)
            loss.backward()
            optimizer.step()
            if CUDA:
                loss = loss.cpu()
            epoch_elbo -= loss.item()
    return epoch_elbo / N, label_mask


def test(data, enc, dec, infer=True):
    enc.eval()
    dec.eval()
    epoch_elbo = 0.0
    epoch_correct = 0
    N = 0
    for b, (images, labels) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += args.batch_size
            if CUDA:
                images = images.cuda()
            q = enc(images, num_samples=NUM_SAMPLES)
            p = dec(images, q, num_samples=NUM_SAMPLES)
            batch_elbo = elbo(q, p)
            if CUDA:
                batch_elbo = batch_elbo.cpu()
            epoch_elbo += batch_elbo.item()
            if infer:
                log_p = p.log_joint(0, 1)
                log_q = q.log_joint(0, 1)
                log_w = log_p - log_q
                w = torch.nn.functional.softmax(log_w, 0)
                y_samples = q['digits'].value
                y_expect = (w.unsqueeze(-1) * y_samples).sum(0)
                _, y_pred = y_expect.max(-1)
                if CUDA:
                    y_pred = y_pred.cpu()
                epoch_correct += (labels == y_pred).sum().item()
            else:
                _, y_pred = q['digits'].value.max(-1)
                if CUDA:
                    y_pred = y_pred.cpu()
                epoch_correct += (labels == y_pred).sum().item() / (NUM_SAMPLES or 1.0)
    return epoch_elbo / N, epoch_correct / N


def get_paired_data(paired_cnt, seed):
    data = torch.utils.data.DataLoader(
        datasets.SVHN(DATA_PATH, split='train', download=True,
                      transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=False)
    tr_labels = data.dataset.labels

    cnt = int(paired_cnt / 10)
    assert cnt == paired_cnt / 10

    label_idx = {}
    for i in range(10):
        label_idx.update({i: []})
    for idx in range(len(tr_labels)):
        label = tr_labels[idx]
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


import time

if args.ckpt_epochs > 0:
    if CUDA:
        enc.load_state_dict(torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs)))
        dec.load_state_dict(torch.load('%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs)))
    else:
        enc.load_state_dict(torch.load('%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
                                       map_location=torch.device('cpu')))
        dec.load_state_dict(torch.load('%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs),
                                       map_location=torch.device('cpu')))
        # enc.load_state_dict(torch.load(args.ckpt_path + '/mnist-semisupervised-50dim-0.0+5a2c637-1.2.0-enc.rar',
        #                                 map_location=torch.device('cpu')))
        # dec.load_state_dict(torch.load(args.ckpt_path+'/mnist-semisupervised-50dim-0.0+5a2c637-1.2.0-dec.rar',
        #                                 map_location=torch.device('cpu')))

mask = {}
fixed_imgs = None
fixed_labels = None
if args.label_frac > 1:
    fixed_imgs, fixed_labels = get_paired_data(args.label_frac, args.seed)

for e in range(args.ckpt_epochs, args.epochs):
    train_start = time.time()
    train_elbo, mask = train(train_data, enc, dec,
                             optimizer, mask, fixed_imgs=fixed_imgs, fixed_labels=fixed_labels)
    train_end = time.time()
    test_start = time.time()
    test_elbo, test_accuracy = test(test_data, enc, dec)
    test_end = time.time()
    print('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e, Accuracy %0.3f (%ds)' % (
        e, train_elbo, train_end - train_start,
        test_elbo, test_accuracy, test_end - test_start))

if not os.path.isdir(args.ckpt_path):
    os.mkdir(args.ckpt_path)
torch.save(enc.state_dict(),
           '%s/%s-encA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.epochs))
torch.save(dec.state_dict(),
           '%s/%s-decA_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.epochs))

if args.ckpt_epochs == args.epochs:
    # util.evaluation.mutual_info(test_data, enc, CUDA, flatten_pixel=NUM_PIXELS, baseline=True)
    util.evaluation.save_traverse_base(args.epochs, test_data, enc, dec, CUDA,
                                       fixed_idxs=[3, 2, 1, 32, 4, 23, 21, 36, 61, 99], output_dir_trvsl=MODEL_NAME,
                                       flatten_pixel=NUM_PIXELS)
    #
    # util.evaluation.save_cross_mnist_base(e, dec, enc, 64,
    #                                           CUDA, MODEL_NAME)
