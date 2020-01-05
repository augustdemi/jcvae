import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import sys

sys.path.append('../')
import probtorch
from probtorch.util import expand_inputs, normal_init, kaiming_init
from torch.nn import functional as F

EPS = 1e-9
TEMP = 0.66


class EncoderA(nn.Module):
    def __init__(self, seed,
                 zShared_dim=18,
                 zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zPrivate_dim = zPrivate_dim
        self.zShared_dim = zShared_dim
        self.seed = seed

        self.enc_hidden = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 2 * zPrivate_dim + 2 * zShared_dim))

        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    def forward(self, x, num_samples=None, q=None):
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
        # attributes
        for i in range(self.zShared_dim):
            q.concrete(logits=shared_logit[:, :, i * 2:(i + 1) * 2],
                       temperature=self.digit_temp,
                       name='sharedA' + str(i))
        return q


class DecoderA(nn.Module):
    def __init__(self, seed,
                 zShared_dim=18,
                 zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP

        self.style_mean = zPrivate_dim
        self.style_std = zPrivate_dim
        self.num_digits = zShared_dim
        self.seed = seed

        self.fc = nn.Sequential(
            nn.Linear(zPrivate_dim + 2 * zShared_dim, 256 * 5 * 5),
            nn.ReLU()
        )

        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    def forward(self, images, shared, q=None, p=None, num_samples=None):
        priv_mean = torch.zeros_like(q['privateA'].dist.loc)
        priv_std = torch.ones_like(q['privateA'].dist.scale)

        p = probtorch.Trace()

        # prior for z_private
        zPrivate = p.normal(priv_mean,
                            priv_std,
                            value=q['privateA'],
                            name='privateA')

        for shared_from in shared.keys():
            latents = [zPrivate]
            # prior for z_shared_atrr

            for i in range(len(shared[shared_from])):
                shared_name = shared[shared_from][i]
                one_zShared = p.concrete(logits=torch.zeros_like(q[shared_name].dist.logits),
                                         temperature=self.digit_temp,
                                         value=q[shared_name],
                                         name=shared_name)

                if 'poe' in shared_from:
                    latents.append(torch.pow(one_zShared + EPS, 1 / 3))  # (sample)^(1/3):prior, modalA,B
                else:
                    latents.append(one_zShared)

            hiddens = self.fc(torch.cat(latents, -1))
            hiddens = hiddens.view(-1, 256, 5, 5)
            images_mean = self.hallucinate(hiddens)
            images_mean = images_mean.view(images_mean.size(0), -1)
            images = images.view(images.size(0), -1)
            # define reconstruction loss (log prob of bernoulli dist)
            p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                      torch.log(1 - x_hat + EPS) * (1 - x)).sum(-1),
                   images_mean, images, name='images_' + shared_from)
        return p

    def forward2(self, latents, cuda):
        hiddens = self.fc(torch.cat(latents, -1))
        hiddens = hiddens.view(-1, 256, 5, 5)
        images_mean = self.hallucinate(hiddens)
        return images_mean


class EncoderB(nn.Module):
    def __init__(self, seed, num_attr=18,
                 num_hidden=256,
                 zShared_dim=18):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zShared_dim = zShared_dim
        self.seed = seed
        self.enc_hidden = nn.Sequential(
            nn.Linear(num_attr, num_hidden),
            nn.ReLU())

        self.fc = nn.Linear(num_hidden, 2 * zShared_dim)
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    @expand_inputs
    def forward(self, attributes, num_samples=None, q=None):
        if q is None:
            q = probtorch.Trace()
        # attributes = attributes.view(attributes.size(0), -1)
        hiddens = self.enc_hidden(attributes)
        shared_attr_logit = self.fc(hiddens)

        # attributes
        for i in range(self.zShared_dim):
            q.concrete(logits=shared_attr_logit[:, :, i * 2:(i + 1) * 2],
                       temperature=self.digit_temp,
                       name='sharedB' + str(i))
        return q


class DecoderB(nn.Module):
    def __init__(self, seed, num_attr=18,
                 num_hidden=256,
                 zShared_dim=18):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP
        self.num_attr = num_attr
        self.seed = seed

        self.dec_hidden = nn.Sequential(
            nn.Linear(2 * zShared_dim, num_hidden),
            nn.ReLU())
        self.dec_label = nn.Sequential(
            nn.Linear(num_hidden, num_attr))
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    def forward(self, attributes, shared, q=None, p=None, num_samples=None, train=True):
        p = probtorch.Trace()
        acc = 0

        for shared_from in shared.keys():
            latents = []
            # prior for z_shared_atrr
            for i in range(len(shared[shared_from])):
                shared_name = shared[shared_from][i]
                if p[shared_name] is not None:
                    one_zShared = p[shared_name].value
                else:
                    one_zShared = p.concrete(logits=torch.zeros_like(q[shared_name].dist.logits),
                                             temperature=self.digit_temp,
                                             value=q[shared_name],
                                             name=shared_name)

                if 'poe' in shared_from:
                    latents.append(torch.pow(one_zShared + EPS, 1 / 3))  # (sample)^(1/3):prior, modalA,B
                else:
                    latents.append(one_zShared)

            hiddens = self.dec_hidden(torch.cat(latents, -1))
            pred_labels = self.dec_label(hiddens)
            pred_labels = pred_labels.squeeze(0)

            pred_labels = F.logsigmoid(pred_labels + EPS)

            p.loss(
                lambda y_pred, target: F.binary_cross_entropy_with_logits(y_pred, target, reduction='none').sum(dim=1), \
                pred_labels, attributes, name='attr_' + shared_from)
            pred_labels = torch.round(torch.exp(pred_labels))

            if 'cross' in shared_from:
                acc = (pred_labels == attributes).sum() / self.num_attr
                f1 = f1_score(attributes.data.to('cpu'), pred_labels.data.to('cpu'), average="samples")
        return p, acc, f1
