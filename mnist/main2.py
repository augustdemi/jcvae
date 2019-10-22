from torchvision import datasets, transforms
import os
import torch

from model import Encoder, Decoder

import sys
sys.path.append('../')
import probtorch

# model parameters
NUM_PIXELS = 784
NUM_HIDDEN = 256
NUM_DIGITS = 10
NUM_STYLE = 50

# training parameters
NUM_SAMPLES = 8
NUM_BATCH = 128
CKPT_EPOCH=500
NUM_EPOCHS = 600
LABEL_FRACTION = 0.02
LEARNING_RATE = 1e-3
BETA1 = 0.90
EPS = 1e-9
CUDA = torch.cuda.is_available()

# path parameters
MODEL_NAME = 'mnist-semisupervised-%02ddim' % NUM_STYLE
DATA_PATH = '../data'
WEIGHTS_PATH = '../weights'
RESTORE = True

print('probtorch:', probtorch.__version__,
      'torch:', torch.__version__,
      'cuda:', torch.cuda.is_available())


if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)

train_data = torch.utils.data.DataLoader(
                datasets.MNIST(DATA_PATH, train=True, download=True,
                               transform=transforms.ToTensor()),
                batch_size=NUM_BATCH, shuffle=True)
test_data = torch.utils.data.DataLoader(
                datasets.MNIST(DATA_PATH, train=False, download=True,
                               transform=transforms.ToTensor()),
                batch_size=NUM_BATCH, shuffle=True)

def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())

enc = Encoder()
dec = Decoder()
if CUDA:
    enc.cuda()
    dec.cuda()
    cuda_tensors(enc)
    cuda_tensors(dec)

optimizer =  torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),
                              lr=LEARNING_RATE,
                              betas=(BETA1, 0.999))

def elbo(q, p, alpha=0.1):
    if NUM_SAMPLES is None:
        return probtorch.objectives.montecarlo.elbo(q, p, sample_dim=None, batch_dim=0, alpha=alpha)
    else:
        return probtorch.objectives.montecarlo.elbo(q, p, sample_dim=0, batch_dim=1, alpha=alpha)


def train(data, enc, dec, optimizer,
          label_mask={}, label_fraction=LABEL_FRACTION):
    epoch_elbo = 0.0
    enc.train()
    dec.train()
    N = 0
    for b, (images, labels) in enumerate(data):
        if images.size()[0] == NUM_BATCH:
            N += NUM_BATCH
            images = images.view(-1, NUM_PIXELS)
            labels_onehot = torch.zeros(NUM_BATCH, NUM_DIGITS)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = torch.clamp(labels_onehot, EPS, 1-EPS)
            if CUDA:
                images = images.cuda()
                labels_onehot = labels_onehot.cuda()
            optimizer.zero_grad()
            if b not in label_mask:
                label_mask[b] = (random() < label_fraction)
            if label_mask[b]:
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
        if images.size()[0] == NUM_BATCH:
            N += NUM_BATCH
            images = images.view(-1, NUM_PIXELS)
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
                _ , y_pred = y_expect.max(-1)
                if CUDA:
                    y_pred = y_pred.cpu()
                epoch_correct += (labels == y_pred).sum().item()
            else:
                _, y_pred = q['digits'].value.max(-1)
                if CUDA:
                    y_pred = y_pred.cpu()
                epoch_correct += (labels == y_pred).sum().item() / (NUM_SAMPLES or 1.0)
    return epoch_elbo / N, epoch_correct / N

import time
from random import random

if RESTORE:
    enc.load_state_dict(torch.load('%s/%s-%s-%s-enc_epoch%s.rar' % (WEIGHTS_PATH, MODEL_NAME, probtorch.__version__, torch.__version__, CKPT_EPOCH)))
    dec.load_state_dict(torch.load('%s/%s-%s-%s-dec_epoch%s.rar' % (WEIGHTS_PATH, MODEL_NAME, probtorch.__version__, torch.__version__, CKPT_EPOCH)))

mask = {}
for e in range(CKPT_EPOCH, NUM_EPOCHS):
    train_start = time.time()
    train_elbo, mask = train(train_data, enc, dec,
                             optimizer, mask, LABEL_FRACTION)
    train_end = time.time()
    test_start = time.time()
    test_elbo, test_accuracy = test(test_data, enc, dec)
    test_end = time.time()
    print('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e, Accuracy %0.3f (%ds)' % (
            e, train_elbo, train_end - train_start,
            test_elbo, test_accuracy, test_end - test_start))

if not os.path.isdir(WEIGHTS_PATH):
    os.mkdir(WEIGHTS_PATH)
torch.save(enc.state_dict(),
           '%s/%s-%s-%s-enc_epoch%s.rar' % (WEIGHTS_PATH, MODEL_NAME, probtorch.__version__, torch.__version__, NUM_EPOCHS ))
torch.save(dec.state_dict(),
           '%s/%s-%s-%s-dec_epoch%s.rar' % (WEIGHTS_PATH, MODEL_NAME, probtorch.__version__, torch.__version__, NUM_EPOCHS))

print('[encoder] ELBO: %e, ACCURACY: %f' % test(test_data, enc, dec, infer=False))
print('[encoder+inference] ELBO: %e, ACCURACY: %f' % test(test_data, enc, dec, infer=True))
