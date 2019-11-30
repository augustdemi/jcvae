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
    def __init__(self, seed, zShared_dim=10,
                 zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zPrivate_dim = zPrivate_dim
        self.zShared_dim = zShared_dim
        self.seed = seed

        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # we do not use the last fc layer.
        self.resnet = nn.Sequential(*modules)

        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(2048, 2 * zPrivate_dim + 2 * zShared_dim),
            nn.Tanh()
        )

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
        if q is None:
            q = probtorch.Trace()
        hiddens = self.resnet(x)
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

        for i in range(self.zShared_dim):
            q.concrete(logits=shared_logit[:, :, 2 * i:2 * (i + 1)],
                       temperature=self.digit_temp,
                       name='sharedA' + str(i))
        return q


class DecoderA(nn.Module):
    def __init__(self, seed,
                 zShared_dim=25,
                 zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP

        self.style_mean = zPrivate_dim
        self.style_std = zPrivate_dim
        self.seed = seed

        self.fc = nn.Sequential(
            nn.Linear(zPrivate_dim + 2 * zShared_dim, 2048),
            nn.ReLU()
        )

        self.layers = nn.ModuleList()

        n_modules = [3, 4, 6, 4]  # num of modules in ResNet50 for seq 1,2,3,4
        n_channels = [[256, 64, 64, 256], [512, 128, 128, 512], [1024, 256, 256, 1024], [2048, 512, 512, 2048]]
        kernel_size = [1, 3, 1]

        # avgpool
        self.layers.append(nn.Sequential(nn.Upsample(scale_factor=4)))

        # en_seq2~ en_seq4
        for i in range(3, 0, -1):
            n_module = n_modules[i]
            n_channel = n_channels[i]
            n = 0
            while n < n_module:
                if n == n_module - 1:
                    n_channel[-1] = n_channel[-1] // 2
                    self.layers.append(nn.Sequential(ResBlk(kernel_size, n_channel, upsample=True)))
                else:
                    self.layers.append(nn.Sequential(ResBlk(kernel_size, n_channel)))
                n += 1

        # seq1
        n_module = n_modules[0]
        n_channel = n_channels[0]
        n = 0
        while n < n_module - 1:
            self.layers.append(nn.Sequential(ResBlk(kernel_size, n_channel)))
            n += 1

        # last module of seq 1
        self.layers.append(nn.Sequential(ResBlk(kernel_size, [256, 64, 64, 64])))

        # maxpool
        self.layers.append(nn.Sequential(nn.Upsample(scale_factor=2)))

        # first conv
        conv_list = [nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
                     nn.BatchNorm2d(3), nn.Sigmoid()]
        self.layers.append(nn.Sequential(*conv_list))

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

        # private은 sharedA(infA), sharedB(crossA), sharedPOE 모두에게 공통적으로 들어가는 node로 z_private 한 샘플에 의해 모두가 다 생성돼야함
        for shared_name in shared.keys():
            latents = [zPrivate]
            # prior for z_shared
            for i in range(len(shared[shared_name])):
                one_zShared = p.concrete(logits=torch.zeros_like(q['sharedA1'].dist.logits),
                                         temperature=self.digit_temp,
                                         value=shared[shared_name][i],
                                         name=shared_name + str(i))

                if 'poe' in shared_name:
                    latents.append(torch.pow(one_zShared + EPS, 1 / 3))  # (sample)^(1/3)
                else:
                    latents.append(one_zShared)

            hiddens = self.fc(torch.cat(latents, -1))
            x = hiddens.view(-1, 2048, 1, 1)
            for layer in self.layers:
                x = layer(x)

            images_mean = x.view(x.size(0), -1)
            images = images.view(images.size(0), -1)
            # define reconstruction loss (log prob of bernoulli dist)
            p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                      torch.log(1 - x_hat + EPS) * (1 - x)).sum(-1),
                   images_mean, images, name='images_' + shared_name)
        return p

    def make_one_hot(self, alpha, cuda):
        _, max_alpha = torch.max(alpha, dim=1)
        one_hot_samples = torch.zeros(alpha.size())
        one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
        if cuda:
            one_hot_samples = one_hot_samples.cuda()
        return one_hot_samples

    def forward2(self, zPrivate, zShared, cuda):
        zShared = self.make_one_hot(zShared.squeeze(0), cuda).unsqueeze(0)
        hiddens = self.dec_hidden(torch.cat([zPrivate, zShared], -1))
        hiddens = hiddens.view(-1, 256, 2, 2)
        images_mean = self.dec_image(hiddens)
        return images_mean


class EncoderB(nn.Module):
    def __init__(self, seed,
                 zShared_dim=25, zPrivate_dim=60):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zPrivate_dim = zPrivate_dim
        self.zShared_dim = zShared_dim
        self.seed = seed
        self.enc_hidden = nn.Sequential(
            nn.Linear(zPrivate_dim + zShared_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.fc = nn.Linear(512, 2 * zPrivate_dim + 2 * zShared_dim)
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
        stats = self.fc(hiddens)

        private_logit = stats[:, :, :2 * self.zPrivate_dim]
        shared_logit = stats[:, :, 2 * self.zPrivate_dim:]

        for i in range(self.zPrivate_dim):
            q.concrete(logits=private_logit[:, :, 2 * i:2 * (i + 1)],
                       temperature=self.digit_temp,
                       name='privateB' + str(i))

        for i in range(self.zShared_dim):
            q.concrete(logits=shared_logit[:, :, 2 * i:2 * (i + 1)],
                       temperature=self.digit_temp,
                       name='sharedB' + str(i))
        return q


class DecoderB(nn.Module):
    def __init__(self, seed, zPrivate_dim=60,
                 zShared_dim=25):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP
        self.seed = seed
        self.zPrivate_dim = zPrivate_dim

        self.dec_hidden = nn.Sequential(
            nn.Linear(2 * zPrivate_dim + 2 * zShared_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.dec_label = nn.Sequential(
            nn.Linear(512, zPrivate_dim + zShared_dim))
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

        priv_latents = []
        for i in range(self.zPrivate_dim):
            latent = p.concrete(logits=torch.zeros_like(q['privateB0'].dist.logits),
                                temperature=self.digit_temp,
                                value=q['privateB' + str(i)],
                                name='privateB' + str(i))
            priv_latents.append(latent)

        # private은 sharedA(infA), sharedB(crossA), sharedPOE 모두에게 공통적으로 들어가는 node로 z_private 한 샘플에 의해 모두가 다 생성돼야함
        for shared_name in shared.keys():
            latents = priv_latents.copy()
            # prior for z_shared # prior is the concrete dist for uniform dist. with all params=1
            for i in range(len(shared[shared_name])):
                one_zShared = p.concrete(logits=torch.zeros_like(q['sharedB0'].dist.logits),
                                         temperature=self.digit_temp,
                                         value=shared[shared_name][i],
                                         name=shared_name + str(i))

                if 'poe' in shared_name:
                    latents.append(torch.pow(one_zShared + EPS, 1 / 3))  # (sample)^(1/3)
                else:
                    latents.append(one_zShared)

            hiddens = self.dec_hidden(torch.cat(latents, -1))
            pred_labels = self.dec_label(hiddens)
            # define reconstruction loss (log prob of bernoulli dist)
            pred_labels = F.logsigmoid(pred_labels + EPS)
            if train:
                p.loss(lambda y_pred, target: -(target * y_pred + (1 - target) * (1 - y_pred)).sum(-1), \
                       pred_labels, attributes.unsqueeze(0), name='attr_' + shared_name)
            else:
                p.loss(lambda y_pred, target: (1 - (target == y_pred).float()), \
                       pred_labels.max(-1)[1], attributes.max(-1)[1], name='attr_' + shared_name)
        return p


''' introVAE: https://github.com/woxuankai/IntroVAE-Pytorch/blob/758328a64dbe4eb650a0916af8ce64f5df9d07e3/model.py '''


class ResBlk(nn.Module):
    def __init__(self, kernels, chs, upsample=False):
        """
        :param kernels: [1, 3, 3], as [kernel_1, kernel_2, kernel_3]
        :param chs: [ch_in, 64, 64, 64], as [ch_in, ch_out1, ch_out2, ch_out3]
        :return:
        """
        assert len(chs) - 1 == len(kernels), "mismatching between chs and kernels"
        assert all(map(lambda x: x % 2 == 1, kernels)), "odd kernel size only"
        super(ResBlk, self).__init__()

        layers = []
        for idx in range(len(kernels)):
            if upsample and idx == 1:
                layers += [nn.ConvTranspose2d(chs[1], chs[2], 4, stride=2, padding=1),
                           nn.BatchNorm2d(chs[2]), nn.ReLU(inplace=True)]
                # layers += [nn.Conv2d(chs[idx], chs[idx + 1], kernels[idx], padding=kernels[idx] // 2, bias=False),
                #            nn.BatchNorm2d(chs[idx + 1]), nn.ReLU(inplace=True)]
                # layers += [nn.Upsample(scale_factor=2, mode='nearest')]
            else:
                layers += [nn.Conv2d(chs[idx], chs[idx + 1], kernels[idx], padding=kernels[idx] // 2),
                           nn.BatchNorm2d(chs[idx + 1]), nn.ReLU(inplace=True)]
        layers.pop()  # remove last activation
        self.net = nn.Sequential(*layers)

        self.shortcut = nn.Sequential()
        if chs[0] != chs[-1]:  # convert from ch_int to ch_out3
            if upsample:
                self.shortcut.add_module('shortcut_conv', nn.ConvTranspose2d(chs[0], chs[-1], 4, stride=2, padding=1))
            else:
                self.shortcut.add_module('shortcut_conv', nn.Conv2d(chs[0], chs[-1], kernel_size=1, padding=0))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(chs[-1]))
            # if upsample:
            # self.shortcut.add_module('shortcut_up', nn.Upsample(scale_factor=2, mode='nearest'))
            # self.outAct = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        # aa =self.shortcut(x)
        # bb =self.net(x)
        return F.relu(self.shortcut(x) + self.net(x), inplace=True)  # Identity + residual
