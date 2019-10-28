from torchvision import datasets, transforms
import os
import torch

from model import Encoder, Decoder

import sys
sys.path.append('../')
import probtorch

# NUM_HIDDEN1 = 400
# NUM_HIDDEN2 = 200


#------------------------------------------------
# training parameters

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=int, default=0, metavar='N',
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
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default: 1e-3]')

    parser.add_argument('--label_frac', type=float, default=0.002,
                        help='how many labels to use')
    parser.add_argument('--sup_frac', type=float, default=0.02,
                        help='supervision ratio')
    parser.add_argument('--lambda_text', type=float, default=10.,
                        help='multipler for text reconstruction [default: 10]')
    parser.add_argument('--beta', type=float, default=10.,
                        help='multipler for TC [default: 10]')

    parser.add_argument('--ckpt_path', type=str, default='../weights',
                        help='save and load path for ckpt')

    args = parser.parse_args()

#------------------------------------------------


EPS = 1e-9
CUDA = torch.cuda.is_available()

# path parameters
MODEL_NAME = 'mnist-run_id%d-priv%02ddim-label_frac%s-sup_frac%s' % (args.run_id, args.n_private, args.label_frac, args.sup_frac)
DATA_PATH = '../data'

import os
desc_file = os.path.join(args.ckpt_path, 'run_id' + str(args.run_id) + '.txt')
with open(desc_file, 'w') as outfile:
    outfile.write(args.run_desc)

BETA = (1., args.beta, 1.)
BIAS_TRAIN = (60000 - 1) / (args.batch_size - 1)
BIAS_TEST = (10000 - 1) / (args.batch_size - 1)
# model parameters
NUM_PIXELS = 784
## model 바꿔서 돌려보기 as marginals.
NUM_HIDDEN = 256
NUM_SAMPLES = 1
if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)

train_data = torch.utils.data.DataLoader(
                datasets.MNIST(DATA_PATH, train=True, download=True,
                               transform=transforms.ToTensor()),
                batch_size=args.batch_size, shuffle=True)
test_data = torch.utils.data.DataLoader(
                datasets.MNIST(DATA_PATH, train=False, download=True,
                               transform=transforms.ToTensor()),
                batch_size=args.batch_size, shuffle=False)

def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())

enc = Encoder(num_pixels=784, num_hidden=256, zShared_dim=10, zPrivate_dim=args.n_private)
dec = Decoder(num_pixels=784, num_hidden=256, zShared_dim=10, zPrivate_dim=args.n_private)
if CUDA:
    enc.cuda()
    dec.cuda()
    cuda_tensors(enc)
    cuda_tensors(dec)

optimizer =  torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),
                              lr=args.lr)


def elbo(iter, q, p, lamb=1.0, beta=(1.0, 1.0, 1.0), bias=1.0, prt=False):
    # from each of modality
    reconst_loss_A, kl_A = probtorch.objectives.mws_tcvae.elbo(q, p, p['imagesA'], latents=['privateA', 'sharedA'], sample_dim=0, batch_dim=1,
                                        lamb=1.0, beta=beta, bias=bias)

    if q['poe'] is not None:
        # by POE
        # 기대하는바:sharedA가 sharedB를 따르길. 즉 sharedA에만 digit정보가 있으며, 그 permutataion이 GT에서처럼 identity이기를
        reconst_loss_poeA, kl_poeA = probtorch.objectives.mws_tcvae.elbo(q, p, p['images_poe'], latents=['privateA', 'poe'], sample_dim=0, batch_dim=1,
                                                    lamb=1.0, beta=beta, bias=bias)
        # 의미 없음. rec 은 항상 0. 인풋이 항상 GT고 poe결과도 GT를 따라갈 확률이 크기 때문(학습 초반엔 A가 unif이라서, 학습 될수록 A가 B label을 잘 따를테니)
        #loss 값 변화 자체로는 의미 없지만, 이 일정한 loss(GT)가 나오도록하는 sharedA의 back pg에는 의미가짐
        reconst_loss_poeB, kl_poeB = probtorch.objectives.mws_tcvae.elbo(q, p, q['labels_poe'], latents=['poe'], sample_dim=0, batch_dim=1,
                                                    lamb=lamb, beta=beta, bias=bias)

        # by cross
        reconst_loss_crA, kl_crA = probtorch.objectives.mws_tcvae.elbo(q, p, p['images_cross'], latents=['privateA', 'sharedB'], sample_dim=0, batch_dim=1,
                                                    lamb=1.0, beta=beta, bias=bias)
        reconst_loss_crB, kl_crB = probtorch.objectives.mws_tcvae.elbo(q, p, q['labels_cross'], latents=['sharedA'], sample_dim=0, batch_dim=1,
                                                    lamb=lamb, beta=beta, bias=bias)

        loss = (reconst_loss_A - kl_A) + (reconst_loss_poeA - kl_poeA) + (lamb * reconst_loss_poeB - kl_poeB) + (reconst_loss_crA - kl_crA) + (lamb * reconst_loss_crB - kl_crB)
        loss = (reconst_loss_poeA - kl_poeA) + (lamb * reconst_loss_poeB - kl_poeB)
        loss = (reconst_loss_poeA - kl_poeA) + (lamb * reconst_loss_poeB - kl_poeB) + (reconst_loss_crA - kl_crA) + (lamb * reconst_loss_crB - kl_crB)
        if iter % 100 == 0:
            print('=========================================')
            print(iter)
            print('reconst_loss_A: ', reconst_loss_A)
            print('kl_A: ', kl_A)
            print('-----------------------------------------')
            print('reconst_loss_poeA: ', reconst_loss_poeA)
            print('reconst_loss_poeA: ', kl_poeA)
            print('-----------------------------------------')
            print('reconst_loss_poeB: ', reconst_loss_poeB)
            print('kl_poeB: ', kl_poeB)
            print('-----------------------------------------')
            print('reconst_loss_crA: ', reconst_loss_crA)
            print('kl_crA: ', kl_crA)
            print('-----------------------------------------')
            print('reconst_loss_crB: ', reconst_loss_crB)
            print('kl_crB: ', kl_crB)
            print('=========================================')
    else:
        loss = 3 * (reconst_loss_A - kl_A)
    return loss

LOSS = 0

def train(data, enc, dec, optimizer,
          label_mask={}, fixed_imgs=None, fixed_labels=None):
    epoch_elbo = 0.0
    enc.train()
    dec.train()
    N = 0
    torch.autograd.set_detect_anomaly(True)
    for b, (images, labels) in enumerate(data):
        if images.size()[0] == args.batch_size:
            N += args.batch_size
            images = images.view(-1, NUM_PIXELS)
            labels_onehot = torch.zeros(args.batch_size, args.n_shared)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = torch.clamp(labels_onehot, EPS, 1-EPS)
            if CUDA:
                images = images.cuda()
                labels_onehot = labels_onehot.cuda()
            optimizer.zero_grad()
            if b not in label_mask:
                label_mask[b] = (random() < args.label_frac)
            if label_mask[b]:
                if args.label_frac == args.sup_frac:
                    q = enc(images, labels_onehot, num_samples=NUM_SAMPLES)
                    p = dec(images, {'private': 'privateA', 'shared': 'sharedA'}, out_name='imagesA', q=q,
                            num_samples=NUM_SAMPLES)
                    p = dec(images, {'private': 'privateA', 'shared': 'sharedB'}, out_name='images_cross', q=q, p=p,
                            num_samples=NUM_SAMPLES)
                    p = dec(images, {'private': 'privateA', 'shared': 'poe'}, out_name='images_poe', q=q, p=p,
                            num_samples=NUM_SAMPLES)
                    loss = -elbo(b, q, p, lamb=args.lambda_text, beta=BETA, bias=BIAS_TRAIN)
            else:
                q = enc(images, num_samples=NUM_SAMPLES)
                p = dec(images, {'private': 'privateA', 'shared': 'sharedA'}, out_name='imagesA', q=q,
                        num_samples=NUM_SAMPLES)
                loss = -elbo(b, q, p, lamb=args.lambda_text, beta=BETA, bias=BIAS_TRAIN)

            if args.label_frac < args.sup_frac and random() < args.sup_frac:
                # print(b)
                N += args.batch_size
                images = fixed_imgs.view(-1, NUM_PIXELS)
                labels_onehot = torch.zeros(args.batch_size, args.n_shared)
                labels_onehot.scatter_(1, fixed_labels.unsqueeze(1), 1)
                labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS)
                optimizer.zero_grad()
                if CUDA:
                    images = images.cuda()
                    labels_onehot = labels_onehot.cuda()

                q = enc(images, labels_onehot, num_samples=NUM_SAMPLES)
                p = dec(images, {'private': 'privateA', 'shared': 'sharedA'}, out_name='imagesA', q=q,
                        num_samples=NUM_SAMPLES)
                p = dec(images, {'private': 'privateA', 'shared': 'sharedB'}, out_name='images_cross', q=q, p=p,
                        num_samples=NUM_SAMPLES)
                p = dec(images, {'private': 'privateA', 'shared': 'poe'}, out_name='images_poe', q=q, p=p,
                        num_samples=NUM_SAMPLES)
                sup_loss = -elbo(b, q, p, lamb=args.lambda_text, beta=BETA, bias=BIAS_TRAIN)
                sup_loss.backward()
                optimizer.step()

                if CUDA:
                    sup_loss = sup_loss.cpu()
                if CUDA:
                    loss = loss.cpu()
                print('-------------- b: ', b)
                print('unsup: ', loss)
                print('sup: ', sup_loss)
                epoch_elbo -= sup_loss.item()
            else:
                loss.backward()
                optimizer.step()
                if CUDA:
                    loss = loss.cpu()
                LOSS = loss
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
            images = images.view(-1, NUM_PIXELS)
            if CUDA:
                images = images.cuda()
            q = enc(images, num_samples=NUM_SAMPLES)
            p = dec(images, {'private': 'privateA', 'shared': 'sharedA'}, out_name='imagesA', q=q,
                    num_samples=NUM_SAMPLES)
            batch_elbo = elbo(b, q, p, lamb=args.lambda_text, beta=BETA, bias=BIAS_TEST)
            if CUDA:
                batch_elbo = batch_elbo.cpu()
            epoch_elbo += batch_elbo.item()
            if infer:
                log_p = p.log_joint(0, 1)
                log_q = q.log_joint(0, 1)
                log_w = log_p - log_q
                w = torch.nn.functional.softmax(log_w, 0)
                y_samples = q['sharedA'].value
                y_expect = (w.unsqueeze(-1) * y_samples).sum(0)
                _ , y_pred = y_expect.max(-1)
                if CUDA:
                    y_pred = y_pred.cpu()
                epoch_correct += (labels == y_pred).sum().item()
            else:
                _, y_pred = q['sharedA'].value.max(-1)
                if CUDA:
                    y_pred = y_pred.cpu()
                epoch_correct += (labels == y_pred).sum().item() / (NUM_SAMPLES or 1.0)
    return epoch_elbo / N, epoch_correct / N


def get_paired_data(paired_cnt):
    data = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_PATH, train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=False)
    per_idx_img = {}
    for i in range(10):
        per_idx_img.update({i:[]})
    for (images, labels) in data:
        for i in range(labels.shape[0]):
            label = int(labels[i].data.detach().cpu().numpy())
            if len(per_idx_img[label]) < int(paired_cnt/10):
                per_idx_img[label].append(images[i])

    imgs = []
    labels = []
    for i in range(10):
        imgs.extend(per_idx_img[i])
        labels.extend([i]*int(paired_cnt/10))
    import numpy as np
    np.random.seed(0)
    np.random.shuffle(imgs)
    np.random.seed(0)
    np.random.shuffle(labels)
    imgs=torch.stack(imgs)
    labels=torch.tensor(labels)
    return imgs, labels



import time
from random import random

if args.ckpt_epochs > 0:
    enc.load_state_dict(torch.load('%s/%s-enc_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs)))
    dec.load_state_dict(torch.load('%s/%s-dec_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.ckpt_epochs)))

mask = {}

if args.label_frac < args.sup_frac:
    fixed_imgs, fixed_labels = get_paired_data(100)


for e in range(args.ckpt_epochs, args.epochs):
    train_start = time.time()
    if args.label_frac < args.sup_frac:
        train_elbo, mask = train(train_data, enc, dec,
                                 optimizer, mask, fixed_imgs=fixed_imgs, fixed_labels=fixed_labels)
    else:
        train_elbo, mask = train(train_data, enc, dec,
                                 optimizer, mask)
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
           '%s/%s-enc_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.epochs ))
torch.save(dec.state_dict(),
           '%s/%s-dec_epoch%s.rar' % (args.ckpt_path, MODEL_NAME, args.epochs))

print('[encoder] ELBO: %e, ACCURACY: %f' % test(test_data, enc, dec, infer=False))
print('[encoder+inference] ELBO: %e, ACCURACY: %f' % test(test_data, enc, dec, infer=True))
