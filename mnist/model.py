import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append('../')
import probtorch
from probtorch.util import expand_inputs, normal_init
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

        self.fc  = nn.Linear(num_hidden, 2*zPrivate_dim + zShared_dim)



    @expand_inputs
    def forward(self, x, num_samples=None, q=None):
        if q is None:
            q = probtorch.Trace()

        hiddens = self.enc_hidden(x)
        stats = self.fc(hiddens)

        muPrivate = stats[:, :, :self.zPrivate_dim]
        logvarPrivate = stats[:, :, self.zPrivate_dim:(2 * self.zPrivate_dim)]
        stdPrivate = torch.sqrt(torch.exp(logvarPrivate) + EPS)

        shared_logit = stats[:, :, (2 * self.zPrivate_dim):]

        q.normal(loc=muPrivate,
                 scale=stdPrivate,
                 name='privateA')
        q.concrete(logits=shared_logit,
                    temperature=self.digit_temp,
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
        digit_log_weights = torch.zeros_like(q['sharedA'].dist.logits) # prior is the concrete dist for uniform dist. with all params=1
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
            zShared = p.concrete(logits=digit_log_weights,
                                temperature=self.digit_temp,
                                value=shared[shared_name],
                                name=shared_name)

            if 'poe' in shared_name:
                hiddens = self.dec_hidden(torch.cat([zPrivate, torch.pow(zShared + EPS, 1/3)], -1))
            else:
                hiddens = self.dec_hidden(torch.cat([zPrivate, zShared], -1))

            images_mean = self.dec_image(hiddens)

            # define reconstruction loss (log prob of bernoulli dist)
            p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                      torch.log(1 - x_hat + EPS) * (1-x)).sum(-1),
                   images_mean, images, name= 'images_' + shared_name)
        return p

    def forward2(self, zPrivate, zShared):
            hiddens = self.dec_hidden(torch.cat([zPrivate, zShared], -1))
            images_mean = self.dec_image(hiddens)
            return images_mean



class EncoderB(nn.Module):
    def __init__(self, num_digis=10,
                       num_hidden=256,
                       zShared_dim=10):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zShared_dim = zShared_dim

        self.enc_hidden = nn.Sequential(
            nn.Linear(num_digis, num_hidden),
            nn.ReLU())

        self.fc  = nn.Linear(num_hidden, zShared_dim)



    @expand_inputs
    def forward(self, labels, num_samples=None, q=None):
        if q is None:
            q = probtorch.Trace()
        hiddens = self.enc_hidden(labels)
        shared_logit = self.fc(hiddens)

        q.concrete(logits=shared_logit,
                    temperature=self.digit_temp,
                    name='sharedB')
        return q



class DecoderB(nn.Module):
    def __init__(self, num_digits=10,
                       num_hidden=256,
                    zShared_dim=10):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP
        self.num_digits = zShared_dim

        self.dec_hidden = nn.Sequential(
                            nn.Linear(zShared_dim, num_hidden),
                            nn.ReLU())
        self.dec_label = nn.Sequential(
                           nn.Linear(num_hidden, num_digits))

    def forward(self, labels, shared, q=None, p=None, num_samples=None, train=True):
        p = probtorch.Trace()
        # private은 sharedA(infA), sharedB(crossA), sharedPOE 모두에게 공통적으로 들어가는 node로 z_private 한 샘플에 의해 모두가 다 생성돼야함
        for shared_name in shared.keys():
            # prior for z_shared # prior is the concrete dist for uniform dist. with all params=1
            zShared = p.concrete(logits=torch.zeros_like(q['sharedB'].dist.logits),
                                temperature=self.digit_temp,
                                value=shared[shared_name],
                                name=shared_name)

            if 'poe' in shared_name:
                hiddens = self.dec_hidden(torch.pow(zShared + EPS, 1/3))
            else:
                hiddens = self.dec_hidden(zShared)

            pred_labels = self.dec_label(hiddens)
            # define reconstruction loss (log prob of bernoulli dist)
            pred_labels = F.log_softmax(pred_labels + EPS, dim=2)
            if train:
                p.loss(lambda y_pred, target: -(target * y_pred).sum(-1), \
                       pred_labels, labels.unsqueeze(0), name='labels_' + shared_name)
            else:
                p.loss(lambda y_pred, target: (1 - (target == y_pred).float()), \
                       pred_labels.max(-1)[1], labels.max(-1)[1], name='labels_' + shared_name)

        return p

