import numpy as np
import torch
import torch.nn as nn

import sys

sys.path.append('../')
import probtorch
from probtorch.util import expand_inputs, normal_init, kaiming_init
from torch.nn import functional as F
from torchvision import models

EPS = 1e-9
TEMP = 0.66


class EncoderA(nn.Module):
    def __init__(self, seed, zPrivate_dim, zSharedAttr_dim, num_hidden):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zPrivate_dim = zPrivate_dim
        self.zSharedAttr_dim = zSharedAttr_dim
        self.seed = seed

        self.fc = nn.Sequential(
            nn.Linear(2048, num_hidden),  # 1560: as CADA_VAE use 1560 hidden
            nn.ReLU(),
            nn.Linear(num_hidden, 2 * zPrivate_dim + 2 * sum(zSharedAttr_dim)),
            nn.ReLU()
        )

        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if m == 'resnet':
                continue
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    # @expand_inputs
    def forward(self, x, num_samples=None, q=None):
        if q is None:
            q = probtorch.Trace()
        stats = self.fc(x)
        stats = stats.unsqueeze(0)
        muPrivate = stats[:, :, :self.zPrivate_dim]
        logvarPrivate = stats[:, :, self.zPrivate_dim:(2 * self.zPrivate_dim)]
        stdPrivate = torch.sqrt(torch.exp(logvarPrivate) + EPS)

        shared_attr_logit = stats[:, :, (2 * self.zPrivate_dim):]

        q.normal(loc=muPrivate,
                 scale=stdPrivate,
                 name='privateA')

        # attributes
        for i in range(sum(self.zSharedAttr_dim)):
            q.concrete(logits=shared_attr_logit[:, :, i * 2:(i + 1) * 2],
                       temperature=self.digit_temp,
                       name='sharedA_attr' + str(i))
        return q


class DecoderA(nn.Module):
    def __init__(self, seed, zPrivate_dim, zSharedAttr_dim, num_hidden):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP
        self.zSharedAttr_dim = zSharedAttr_dim

        self.style_mean = zPrivate_dim
        self.style_std = zPrivate_dim
        self.seed = seed

        self.dec_hidden = nn.Sequential(
            nn.Linear(zPrivate_dim + 2 * sum(zSharedAttr_dim), num_hidden),
            nn.ReLU()
        )

        self.dec_image = nn.Sequential(
            nn.Linear(num_hidden, 2048)
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
                one_attr_zShared = p.concrete(logits=torch.zeros(2),
                                              temperature=self.digit_temp,
                                              value=q[shared_name],
                                              name=shared_name)

                if 'poe' in shared_from:
                    latents.append(torch.pow(one_attr_zShared + EPS, 1 / 3))  # (sample)^(1/3):prior, modalA,B
                else:
                    latents.append(one_attr_zShared)

            hiddens = self.dec_hidden(torch.cat(latents, -1))
            pred_imgs = self.dec_image(hiddens)
            pred_imgs = pred_imgs.squeeze(0)
            pred_imgs = F.logsigmoid(pred_imgs + EPS)

            p.loss(
                lambda y_pred, target: F.binary_cross_entropy_with_logits(y_pred, target, reduction='none').sum(dim=1), \
                pred_imgs, images, name='images_' + shared_from)
        return p

    def forward2(self, latents, cuda):
        hiddens = self.fc(torch.cat(latents, -1))
        x = hiddens.view(-1, 2048, 1, 1)
        for layer in self.layers:
            x = layer(x)

        return x


class EncoderB(nn.Module):
    def __init__(self, seed, zSharedAttr_dim, num_hidden):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zSharedAttr_dim = zSharedAttr_dim
        self.seed = seed
        self.enc_hidden = nn.Sequential(
            nn.Linear(sum(zSharedAttr_dim), num_hidden),
            nn.ReLU(),
        )

        self.fc = nn.Linear(num_hidden, sum(zSharedAttr_dim) * 2)
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

        for i in range(sum(self.zSharedAttr_dim)):
            q.concrete(logits=shared_attr_logit[:, :, i * 2:(i + 1) * 2],
                       temperature=self.digit_temp,
                       name='sharedB_attr' + str(i))
        return q


class DecoderB(nn.Module):
    def __init__(self, seed, zSharedAttr_dim, num_hidden):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP
        self.seed = seed
        self.zSharedAttr_dim = zSharedAttr_dim

        self.dec_hidden = nn.Sequential(
            nn.Linear(2 * sum(zSharedAttr_dim), num_hidden),
            nn.ReLU(),
        )
        self.dec_label = nn.Sequential(
            nn.Linear(num_hidden, sum(zSharedAttr_dim)))
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

        for shared_from in shared.keys():
            latents = []
            # prior for z_shared_atrr
            for i in range(len(shared[shared_from])):
                shared_name = shared[shared_from][i]
                if p[shared_name] is not None:
                    one_attr_zShared = p[shared_name].value
                else:
                    one_attr_zShared = p.concrete(logits=torch.zeros(2),
                                                  temperature=self.digit_temp,
                                                  value=q[shared_name],
                                                  name=shared_name)

                if 'poe' in shared_from:
                    latents.append(torch.pow(one_attr_zShared + EPS, 1 / 3))  # (sample)^(1/3):prior, modalA,B
                else:
                    latents.append(one_attr_zShared)

            hiddens = self.dec_hidden(torch.cat(latents, -1))
            pred_labels = self.dec_label(hiddens)
            pred_labels = pred_labels.squeeze(0)

            pred_labels = F.logsigmoid(pred_labels + EPS)

            p.loss(
                lambda y_pred, target: F.binary_cross_entropy_with_logits(y_pred, target, reduction='none').sum(dim=1), \
                pred_labels, attributes, name='attr_' + shared_from)
            pred_labels = torch.round(torch.exp(pred_labels))
            acc = (pred_labels == attributes).sum() / sum(self.zSharedAttr_dim)
        return p, acc
