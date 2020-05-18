import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append('../')
import probtorch
from probtorch.util import expand_inputs, kaiming_init
from torch.nn import functional as F

EPS = 1e-9
TEMP = 0.66

class EncoderA(nn.Module):
    def __init__(self, seed,
                 zShared_dim=10,
                 zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.seed = seed
        self.digit_temp = torch.tensor(TEMP)
        self.zPrivate_dim = zPrivate_dim
        self.zShared_dim = zShared_dim

        self.enc_hidden = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * zPrivate_dim + zShared_dim))
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)


    # @expand_inputs
    def forward(self, x, num_samples=None, q=None):
        # print(self._modules['enc_hidden'][0].weight.sum())
        if q is None:
            q = probtorch.Trace()

        hiddens = self.enc_hidden(x)
        hiddens = hiddens.view(hiddens.size(0), -1)
        stats = self.fc(hiddens)
        stats = stats.unsqueeze(0)
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
    def __init__(self, seed,
                 zShared_dim=10,
                 zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP
        self.seed = seed

        self.dec_hidden = nn.Sequential(
            nn.Linear(zPrivate_dim + zShared_dim, 512),
                            nn.ReLU(),
            nn.Linear(512, 128 * 7 * 7),
                            nn.ReLU())
        self.dec_image = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                           nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
                           nn.Sigmoid())
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)


    def forward(self, images, shared, q=None, p=None, num_samples=None):
        digit_log_weights = torch.zeros_like(q['sharedA'].dist.logits)
        private_mean = torch.zeros_like(q['privateA'].dist.loc)
        private_std = torch.ones_like(q['privateA'].dist.scale)

        p = probtorch.Trace()

        # prior for z_private
        zPrivate = p.normal(private_mean,
                            private_std,
                            value=q['privateA'],
                            name='privateA')

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

            hiddens = hiddens.view(-1, 128, 7, 7)
            images_mean = self.dec_image(hiddens)

            images_mean = images_mean.view(images_mean.size(0), -1)
            images = images.view(images.size(0), -1)
            # define reconstruction loss (log prob of bernoulli dist)
            p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                      torch.log(1 - x_hat + EPS) * (1-x)).sum(-1),
                   images_mean, images, name= 'images_' + shared_name)
        return p


class EncoderB(nn.Module):
    def __init__(self, seed, num_digis=10,
                 num_hidden=512,
                 zShared_dim=10):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zShared_dim = zShared_dim
        self.seed = seed
        self.enc_hidden = nn.Sequential(
            nn.Linear(10, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.fc = nn.Linear(num_hidden, zShared_dim)
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)


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
    def __init__(self, seed, num_digits=10,
                 num_hidden=512,
                 zShared_dim=10):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP
        self.seed = seed

        self.dec_hidden = nn.Sequential(
            nn.Linear(zShared_dim, num_hidden),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.dec_label = nn.Sequential(
            nn.Linear(num_hidden, num_digits))
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

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
                p.loss(lambda y_pred, target: (1 - (target == y_pred).float()), \
                       pred_labels.max(-1)[1], labels.max(-1)[1], name='labels_acc_' + shared_name)
            else:
                p.loss(lambda y_pred, target: (1 - (target == y_pred).float()), \
                       pred_labels.max(-1)[1], labels.max(-1)[1], name='labels_' + shared_name)

        return p

