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
                       num_hidden1=256,
                       num_hidden2=256,
                       zShared_dim=10,
                     zPrivate_dim=50):
        super(self.__class__, self).__init__()

        self.zPrivate_dim = zPrivate_dim
        self.zShared_dim = zShared_dim

        self.enc_hidden = nn.Sequential(
            nn.Linear(num_pixels, num_hidden1),
            nn.ReLU(),
            nn.Linear(num_hidden1, num_hidden2),
            nn.ReLU()
        )

        self.fc  = nn.Linear(num_hidden2, 2*zPrivate_dim + zShared_dim)
        self.digit_temp = torch.tensor(0.66)

    @expand_inputs
    def forward(self, x, labels=None, num_samples=None):
        q = probtorch.Trace()
        hiddens = self.enc_hidden(x)
        stats = self.fc(hiddens)

        muPrivate = stats[:, :, :self.zPrivate_dim]
        logvarPrivate = stats[:, :, self.zPrivate_dim:(2 * self.zPrivate_dim)]
        stdPrivate = torch.exp(logvarPrivate)

        shared_logit = stats[:, :, (2 * self.zPrivate_dim):]

        q.normal(loc=muPrivate,
                 scale=stdPrivate,
                 name='privateA')

        q.concrete(logits=shared_logit,
                    temperature=self.digit_temp,
                    name='sharedA')
        if labels is not None:
            alpha_logit = shared_logit
            beta_logit = torch.log(labels + EPS) # param for concrete dist. should be bigger than 0
            prior_logit = torch.zeros_like(labels) # prior is the concrete dist. for uniform dist.
            poe_logit = torch.pow(alpha_logit + beta_logit + prior_logit, 3)

            q.concrete(logits=poe_logit,
                       temperature=self.digit_temp,
                       name='poe') # will follow sharedB since sharedB is obtained by identity map

            q.concrete(logits=beta_logit,
                       temperature=self.digit_temp,
                       value=labels,
                       name='sharedB')

            label_loss = lambda y_pred, target: (1 - (target == y_pred).float())
            q.loss(label_loss, q['sharedA'].value.max(-1)[1], labels.max(-1)[1], name='labels_cross')
            # print(torch.log( (q['sharedA'].value.max(-1)[1] == labels.max(-1)[1]).float() + 1e-9))
            q.loss(label_loss, q['poe'].value.max(-1)[1], labels.max(-1)[1], name='labels_poe')
        return q


class Decoder(nn.Module):
    def __init__(self, num_pixels=784,
                       num_hidden1=256,
                       num_hidden2=256,
                    zShared_dim=10,
                    zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.digit_temp = 0.66

        self.style_mean = zPrivate_dim
        self.style_std = zPrivate_dim
        self.num_digits = zShared_dim

        self.dec_hidden = nn.Sequential(
                            nn.Linear(zPrivate_dim + zShared_dim, num_hidden2),
                            nn.ReLU(),
                            nn.Linear(num_hidden2, num_hidden1),
                            nn.ReLU())
        self.dec_image = nn.Sequential(
                           nn.Linear(num_hidden1, num_pixels),
                           nn.Sigmoid())

    def forward(self, images, latents, out_name, q=None, p=None, num_samples=None):


        private=latents['private']
        shared=latents['shared']
        digit_log_weights = torch.zeros_like(q[shared].dist.logits) # prior is the concrete dist for uniform dist. with all params=1
        style_mean = torch.zeros_like(q[private].dist.loc)
        style_std = torch.ones_like(q[private].dist.scale)

        if p is None:
            p = probtorch.Trace()
            p.normal(style_mean,
                            style_std,
                            value=q[private],
                            name=private)
        zPrivate = p[private].value
        zShared = p.concrete(logits=digit_log_weights,
                            temperature=self.digit_temp,
                            value=q[shared],
                            name=shared)
        if shared == 'poe':
            hiddens = self.dec_hidden(torch.cat([torch.pow(zShared + EPS, 1/3), zPrivate], -1))
            # hiddens = self.dec_hidden(torch.cat([zShared, zPrivate], -1))
        else:
            hiddens = self.dec_hidden(torch.cat([zShared, zPrivate], -1))
        images_mean = self.dec_image(hiddens)
        # define reconstruction loss (log prob of bernoulli dist)
        p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                  torch.log(1 - x_hat + EPS) * (1-x)).sum(-1),
               images_mean, images, name=out_name)
        return p