import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append('../')
import probtorch
from probtorch.util import expand_inputs
from torch.nn import functional as F

EPS = 1e-9
TEMP = 0.66

class EncoderA(nn.Module):
    def __init__(self, num_pixels=784,
                       num_hidden=256,
                       zShared_dim=10,
                     zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zPrivate_dim = zPrivate_dim
        self.zShared_dim = zShared_dim

        self.enc_hidden = nn.Sequential(
            nn.Linear(num_pixels, num_hidden),
            nn.ReLU())

        self.fc  = nn.Linear(num_hidden, 2*zPrivate_dim + 2*zShared_dim)

    @expand_inputs
    def forward(self, x, num_samples=None, q=None):
        if q is None:
            q = probtorch.Trace()

        hiddens = self.enc_hidden(x)
        stats = self.fc(hiddens)

        muPrivate = stats[:, :, :self.zPrivate_dim]
        logvarPrivate = stats[:, :, self.zPrivate_dim:(2 * self.zPrivate_dim)]
        stdPrivate = torch.sqrt(torch.exp(logvarPrivate) + EPS)


        muShared = stats[:, :, (2 * self.zPrivate_dim):(2 * self.zPrivate_dim + self.zShared_dim)]
        logvarShared = stats[:, :, (2 * self.zPrivate_dim + self.zShared_dim):]
        stdShared = torch.sqrt(torch.exp(logvarShared) + EPS)

        q.normal(loc=muPrivate,
                 scale=stdPrivate,
                 name='privateA')
        q.normal(loc=muShared,
                 scale=stdShared,
                 name='sharedA')
        return q


class DecoderA(nn.Module):
    def __init__(self, num_pixels=784,
                       num_hidden=256,
                    zShared_dim=10,
                    zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP

        self.style_mean = zPrivate_dim
        self.style_std = zPrivate_dim
        self.num_digits = zShared_dim

        self.dec_hidden = nn.Sequential(
                            nn.Linear(zPrivate_dim + zShared_dim, num_hidden),
                            nn.ReLU())
        self.dec_image = nn.Sequential(
                           nn.Linear(num_hidden, num_pixels),
                           nn.Sigmoid())

    def forward(self, images, shared, q=None, p=None, num_samples=None):
        style_mean = torch.zeros_like(q['privateA'].dist.loc)
        style_std = torch.ones_like(q['privateA'].dist.scale)

        p = probtorch.Trace()

        # prior for z_private
        zPrivate = p.normal(style_mean,
                        style_std,
                        value=q['privateA'],
                        name='privateA')
        # private은 sharedA(infA), sharedB(crossA), sharedPOE 모두에게 공통적으로 들어가는 node로 z_private 한 샘플에 의해 모두가 다 생성돼야함
        for shared_name in shared.keys():
            # prior for z_shared
            zShared = p.normal(style_mean,
                                style_std,
                                value=shared[shared_name],
                                name=shared_name)

            hiddens = self.dec_hidden(torch.cat([zPrivate, zShared], -1))
            images_mean = self.dec_image(hiddens)

            # define reconstruction loss (log prob of bernoulli dist)
            p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                      torch.log(1 - x_hat + EPS) * (1-x)).sum(-1),
                   images_mean, images, name= 'imagesA_' + shared_name)
        return p

    def forward2(self, zPrivate, zShared):
            hiddens = self.dec_hidden(torch.cat([zPrivate, zShared], -1))
            images_mean = self.dec_image(hiddens)
            return images_mean

class EncoderB(nn.Module):
    def __init__(self, num_pixels=784,
                       num_hidden=256,
                       zShared_dim=10,
                     zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zPrivate_dim = zPrivate_dim
        self.zShared_dim = zShared_dim

        self.enc_hidden = nn.Sequential(
            nn.Linear(num_pixels, num_hidden),
            nn.ReLU())

        self.fc  = nn.Linear(num_hidden, 2*zPrivate_dim + 2*zShared_dim)

    @expand_inputs
    def forward(self, x, num_samples=None, q=None):
        if q is None:
            q = probtorch.Trace()

        hiddens = self.enc_hidden(x)
        stats = self.fc(hiddens)

        muPrivate = stats[:, :, :self.zPrivate_dim]
        logvarPrivate = stats[:, :, self.zPrivate_dim:(2 * self.zPrivate_dim)]
        stdPrivate = torch.sqrt(torch.exp(logvarPrivate) + EPS)


        muShared = stats[:, :, (2 * self.zPrivate_dim):(2 * self.zPrivate_dim + self.zShared_dim)]
        logvarShared = stats[:, :, (2 * self.zPrivate_dim + self.zShared_dim):]
        stdShared = torch.sqrt(torch.exp(logvarShared) + EPS)

        q.normal(loc=muPrivate,
                 scale=stdPrivate,
                 name='privateB')
        q.normal(loc=muShared,
                 scale=stdShared,
                 name='sharedB')
        return q


class DecoderB(nn.Module):
    def __init__(self, num_pixels=784,
                       num_hidden=256,
                    zShared_dim=10,
                    zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP

        self.style_mean = zPrivate_dim
        self.style_std = zPrivate_dim
        self.num_digits = zShared_dim

        self.dec_hidden = nn.Sequential(
                            nn.Linear(zPrivate_dim + zShared_dim, num_hidden),
                            nn.ReLU())
        self.dec_image = nn.Sequential(
                           nn.Linear(num_hidden, num_pixels),
                           nn.Sigmoid())

    def forward(self, images, shared, q=None, p=None, num_samples=None):
        style_mean = torch.zeros_like(q['privateB'].dist.loc)
        style_std = torch.ones_like(q['privateB'].dist.scale)

        p = probtorch.Trace()

        # prior for z_private
        zPrivate = p.normal(style_mean,
                        style_std,
                        value=q['privateB'],
                        name='privateB')
        # private은 sharedA(infA), sharedB(crossA), sharedPOE 모두에게 공통적으로 들어가는 node로 z_private 한 샘플에 의해 모두가 다 생성돼야함
        for shared_name in shared.keys():
            # prior for z_shared
            zShared = p.normal(style_mean,
                                style_std,
                                value=shared[shared_name],
                                name=shared_name)

            hiddens = self.dec_hidden(torch.cat([zPrivate, zShared], -1))
            images_mean = self.dec_image(hiddens)

            # define reconstruction loss (log prob of bernoulli dist)
            p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                      torch.log(1 - x_hat + EPS) * (1-x)).sum(-1),
                   images_mean, images, name= 'imagesB_' + shared_name)
        return p

    def forward2(self, zPrivate, zShared):
            hiddens = self.dec_hidden(torch.cat([zPrivate, zShared], -1))
            images_mean = self.dec_image(hiddens)
            return images_mean

