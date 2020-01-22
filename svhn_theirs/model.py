import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from torch.nn import Parameter
import torch
import torch.nn as nn

import sys

sys.path.append('../')
import probtorch
from probtorch.util import expand_inputs

# model parameters
# NUM_PIXELS = 784
# NUM_HIDDEN = 256
# NUM_DIGITS = 10
# NUM_STYLE = 50
#
# # training parameters
# NUM_SAMPLES = 8
# NUM_BATCH = 128
# NUM_EPOCHS = 200
# LABEL_FRACTION = 0.01
# LEARNING_RATE = 1e-3
# BETA1 = 0.90
EPS = 1e-9


# #CUDA = torch.cuda.is_available()
# CUDA = True
#
# # path parameters
# MODEL_NAME = 'mnist-semisupervised-%02ddim' % NUM_STYLE
# DATA_PATH = '../data'
# WEIGHTS_PATH = '../weights'
# RESTORE = False




class Encoder(nn.Module):
    def __init__(self,
                 num_digits=10,
                 num_style=50):
        super(self.__class__, self).__init__()
        self.enc_hidden = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.ReLU()
        )
        num_hidden = 256 * 2 * 2
        self.digit_log_weights = nn.Linear(num_hidden, num_digits)
        self.digit_temp = torch.tensor(0.66)
        self.style_mean = nn.Linear(num_hidden + num_digits, num_style)
        self.style_log_std = nn.Linear(num_hidden + num_digits, num_style)

    @expand_inputs
    def forward(self, images, labels=None, num_samples=None):
        q = probtorch.Trace()
        images = images.squeeze(0)
        hiddens = self.enc_hidden(images)
        hiddens = hiddens.view(hiddens.size(0), -1)
        digits = q.concrete(logits=self.digit_log_weights(hiddens).unsqueeze(0),
                            temperature=self.digit_temp,
                            value=labels,
                            name='digits')
        hiddens2 = torch.cat([digits.squeeze(0), hiddens], -1)
        styles_mean = self.style_mean(hiddens2)
        styles_std = torch.exp(self.style_log_std(hiddens2))
        q.normal(styles_mean.unsqueeze(0),
                 styles_std.unsqueeze(0),
                 name='styles')
        return q


class Decoder(nn.Module):
    def __init__(self,
                 num_digits=10,
                 num_style=50):
        super(self.__class__, self).__init__()
        self.num_digits = num_digits
        self.digit_log_weights = torch.zeros(num_digits)
        self.digit_temp = 0.66
        self.style_mean = torch.zeros(num_style)
        self.style_std = torch.ones(num_style)
        num_hidden = 256 * 2 * 2
        self.dec_hidden = nn.Sequential(
            nn.Linear(num_style + num_digits, num_hidden),
            nn.ReLU())
        self.dec_image = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Sigmoid())

    def forward(self, images, q=None, num_samples=None):
        p = probtorch.Trace()
        digits = p.concrete(logits=self.digit_log_weights,
                            temperature=self.digit_temp,
                            value=q['digits'],
                            name='digits')
        styles = p.normal(self.style_mean,
                          self.style_std,
                          value=q['styles'],
                          name='styles')
        hiddens = self.dec_hidden(torch.cat([digits, styles], -1))
        hiddens = hiddens.view(-1, 256, 2, 2)
        images_mean = self.dec_image(hiddens)

        images_mean = images_mean.view(images_mean.size(0), -1).unsqueeze(0)
        images = images.view(images.size(0), -1).unsqueeze(0)
        p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                  torch.log(1 - x_hat + EPS) * (1 - x)).sum(-1),
               images_mean, images, name='images')
        return p

    def make_one_hot(self, alpha, cuda):
        _, max_alpha = torch.max(alpha, dim=1)
        one_hot_samples = torch.zeros(alpha.size())
        one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
        if cuda:
            one_hot_samples = one_hot_samples.cuda()
        return one_hot_samples

    def forward2(self, digits, styles, cuda):
        digits = self.make_one_hot(digits.squeeze(0), cuda).unsqueeze(0)
        hiddens = self.dec_hidden(torch.cat([digits, styles], -1))
        images_mean = self.dec_image(hiddens)
        return images_mean
