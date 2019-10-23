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

EPS = 1e-9


class Encoder(nn.Module):
    def __init__(self, num_pixels=784,
                       num_hidden=256,
                       zShared_dim=10,
                     zPrivate_dim=50):
        super(self.__class__, self).__init__()

        self.zPrivate_dim = zPrivate_dim
        self.zShared_dim = zShared_dim

        self.enc_hidden = nn.Sequential(
            nn.Linear(num_pixels, num_hidden),
            nn.ReLU())

        self.fc  = nn.Linear(num_hidden, 2*zPrivate_dim + zShared_dim)
        self.digit_temp = torch.tensor(0.66)

    @expand_inputs
    def forward(self, x, labels=None, num_samples=None):
        q = probtorch.Trace()
        hiddens = self.enc_hidden(x)
        stats = self.fc(hiddens)

        muPrivate = stats[:, :, :self.zPrivate_dim]
        logvarPrivate = stats[:, :, self.zPrivate_dim:(2 * self.zPrivate_dim)]
        stdPrivate = torch.exp(logvarPrivate)

        q.normal(muPrivate,
                 stdPrivate,
                 name='styles')

        q.concrete(logits=stats[:, :, (2 * self.zPrivate_dim):],
                            temperature=self.digit_temp,
                            value=labels,
                            name='digits')
        return q


class Decoder(nn.Module):
    def __init__(self, num_pixels=784,
                       num_hidden=256,
                    zShared_dim=10,
                    zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.num_digits = zShared_dim
        self.digit_log_weights = torch.zeros(zShared_dim)
        self.digit_temp = 0.66

        self.style_mean = torch.zeros(zPrivate_dim)
        self.style_std = torch.ones(zPrivate_dim)

        self.dec_hidden = nn.Sequential(
                            nn.Linear(zPrivate_dim + zShared_dim, num_hidden),
                            nn.ReLU())
        self.dec_image = nn.Sequential(
                           nn.Linear(num_hidden, num_pixels),
                           nn.Sigmoid())

    def forward(self, images, q=None, num_samples=None):
        p = probtorch.Trace()
        zShared = p.concrete(logits=self.digit_log_weights,
                            temperature=self.digit_temp,
                            value=q['digits'],
                            name='digits')
        zPrivate = p.normal(self.style_mean,
                          self.style_std,
                          value=q['styles'],
                          name='styles')

        hiddens = self.dec_hidden(torch.cat([zShared, zPrivate], -1))
        images_mean = self.dec_image(hiddens)
        # define reconstruction loss (log prob of bernoulli dist)
        p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                  torch.log(1 - x_hat + EPS) * (1-x)).sum(-1),
               images_mean, images, name='images')
        return p