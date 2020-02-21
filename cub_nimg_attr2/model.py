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
N_PIXELS = 3 * 128 * 128
N_ATTR = 312


class EncoderA(nn.Module):
    def __init__(self, seed, zPrivate_dim, zShared_dim, num_hidden):
        super(self.__class__, self).__init__()
        self.zPrivate_dim = zPrivate_dim
        self.zShared_dim = zShared_dim
        self.seed = seed

        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # we do not use the last fc layer.
        self.resnet = nn.Sequential(*modules)

        self.fc = nn.Sequential(
            nn.Linear(2048, 2 * zPrivate_dim + 2 * zShared_dim),
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
        feature = self.resnet(x)
        feature = feature.view(feature.size(0), -1)
        stats = self.fc(feature)
        stats = stats.unsqueeze(0)
        muPrivate = stats[:, :, :self.zPrivate_dim]
        logvarPrivate = stats[:, :, self.zPrivate_dim:(2 * self.zPrivate_dim)]
        stdPrivate = torch.sqrt(torch.exp(logvarPrivate) + EPS)

        muShared = stats[:, :, (2 * self.zPrivate_dim):(2 * self.zPrivate_dim + self.zShared_dim)]
        logvarShared = stats[:, :, (2 * self.zPrivate_dim + self.zShared_dim):]
        stdShared = torch.sqrt(torch.exp(logvarShared) + EPS)

        q.normal(loc=muPrivate,
                 scale=stdPrivate,
                 name='privateA')

        # attributes
        q.normal(loc=muShared,
                 scale=stdShared,
                 name='sharedA')
        return q


# class DecoderA(nn.Module):
#     def __init__(self, seed, zPrivate_dim, zShared_dim, num_hidden):
#         super(self.__class__, self).__init__()
#         self.digit_temp = TEMP
#         self.zShared_dim = zShared_dim
#
#         self.style_mean = zPrivate_dim
#         self.style_std = zPrivate_dim
#         self.seed = seed
#
#         self.fc = nn.Sequential(
#             nn.Linear(zPrivate_dim + zShared_dim, 2048),
#             nn.ReLU()
#         )
#
#         self.layers = nn.ModuleList()
#
#         n_modules = [3, 4, 6, 4]  # num of modules in ResNet50 for seq 1,2,3,4
#         n_channels = [[256, 64, 64, 256], [512, 128, 128, 512], [1024, 256, 256, 1024], [2048, 512, 512, 2048]]
#         kernel_size = [1, 3, 1]
#
#         # avgpool
#         self.layers.append(nn.Sequential(nn.Upsample(scale_factor=4)))
#
#         # en_seq2~ en_seq4
#         for i in range(3, 0, -1):
#             n_module = n_modules[i]
#             n_channel = n_channels[i]
#             n = 0
#             while n < n_module:
#                 if n == n_module - 1:
#                     n_channel[-1] = n_channel[-1] // 2
#                     self.layers.append(nn.Sequential(ResBlk(kernel_size, n_channel, upsample=True)))
#                 else:
#                     self.layers.append(nn.Sequential(ResBlk(kernel_size, n_channel)))
#                 n += 1
#
#         # seq1
#         n_module = n_modules[0]
#         n_channel = n_channels[0]
#         n = 0
#         while n < n_module - 1:
#             self.layers.append(nn.Sequential(ResBlk(kernel_size, n_channel)))
#             n += 1
#
#         # last module of seq 1
#         self.layers.append(nn.Sequential(ResBlk(kernel_size, [256, 64, 64, 64])))
#
#         # maxpool
#         self.layers.append(nn.Sequential(nn.Upsample(scale_factor=2)))
#
#         # first conv
#         conv_list = [nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
#                      nn.BatchNorm2d(3), nn.Sigmoid()]
#         self.layers.append(nn.Sequential(*conv_list))
#
#         self.weight_init()
#
#     def weight_init(self):
#         for m in self._modules:
#             if isinstance(self._modules[m], nn.Sequential):
#                 for one_module in self._modules[m]:
#                     kaiming_init(one_module, self.seed)
#             else:
#                 kaiming_init(self._modules[m], self.seed)
#
#     def forward(self, images, shared, q=None, p=None, num_samples=None):
#         priv_mean = torch.zeros_like(q['privateA'].dist.loc)
#         priv_std = torch.ones_like(q['privateA'].dist.scale)
#         shared_mean = torch.zeros_like(q['sharedA'].dist.loc)
#         shared_std = torch.ones_like(q['sharedA'].dist.scale)
#
#         p = probtorch.Trace()
#
#         # prior for z_private
#         zPrivate = p.normal(priv_mean,
#                             priv_std,
#                             value=q['privateA'],
#                             name='privateA')
#
#         for shared_from in shared.keys():
#             latents = [zPrivate]
#             # prior for z_shared_atrr
#             zShared = p.normal(shared_mean,
#                                shared_std,
#                                value=q[shared[shared_from]],
#                                name=shared[shared_from])
#             latents.append(zShared)
#
#             hiddens = self.fc(torch.cat(latents, -1))
#             x = hiddens.view(-1, 2048, 1, 1)
#             for layer in self.layers:
#                 x = layer(x)
#
#             images_mean = x.view(x.size(0), -1)
#             images = images.view(images.size(0), -1)
#             # define reconstruction loss (log prob of bernoulli dist)
#             p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
#                                       torch.log(1 - x_hat + EPS) * (1 - x)).sum(-1),
#                    images_mean, images, name='images_' + shared_from)
#         return p
#
#     def forward2(self, latents):
#         hiddens = self.fc(torch.cat(latents, -1))
#         x = hiddens.view(-1, 2048, 1, 1)
#         for layer in self.layers:
#             x = layer(x)
#         return x


class DecoderA(nn.Module):
    def __init__(self, seed, zPrivate_dim, zShared_dim, num_hidden):
        super(self.__class__, self).__init__()
        self.zShared_dim = zShared_dim

        self.style_mean = zPrivate_dim
        self.style_std = zPrivate_dim
        self.seed = seed

        self.dec_hidden = nn.Sequential(
            nn.Linear(zPrivate_dim + zShared_dim, 2048),
            nn.ReLU()
        )

        self.dec_image = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
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
        priv_mean = torch.zeros_like(q['privateA'].dist.loc)
        priv_std = torch.ones_like(q['privateA'].dist.scale)
        shared_mean = torch.zeros_like(q['sharedA'].dist.loc)
        shared_std = torch.ones_like(q['sharedA'].dist.scale)
        p = probtorch.Trace()

        # prior for z_private
        zPrivate = p.normal(priv_mean,
                            priv_std,
                            value=q['privateA'],
                            name='privateA')

        for shared_from in shared.keys():
            latents = [zPrivate]
            # prior for z_shared_atrr
            zShared = p.normal(shared_mean,
                               shared_std,
                               value=q[shared[shared_from]],
                               name=shared[shared_from])
            latents.append(zShared)

            hiddens = self.dec_hidden(torch.cat(latents, -1))
            x = hiddens.view(-1, 512, 2, 2)
            images_mean = self.dec_image(x)

            images_mean = images_mean.view(images_mean.size(0), -1)
            images = images.view(images.size(0), -1)
            # define reconstruction loss (log prob of bernoulli dist)
            p.loss(lambda x_hat, x: - ((torch.log(x_hat + EPS) * x +
                                        torch.log(1 - x_hat + EPS) * (1 - x)).sum(-1)) / N_PIXELS,
                   images_mean, images, name='images_' + shared_from)
        return p

    def forward2(self, latents):
        hiddens = self.dec_hidden(torch.cat(latents, -1))
        x = hiddens.view(-1, 512, 2, 2)
        images_mean = self.dec_image(x)
        return images_mean



class EncoderB(nn.Module):
    def __init__(self, seed, zShared_dim, num_hidden):
        super(self.__class__, self).__init__()
        self.zShared_dim = zShared_dim
        self.seed = seed
        # self.enc_hidden = nn.Sequential(
        #     nn.Linear(312, num_hidden),
        #     nn.ReLU(),
        # )

        self.fc = nn.Linear(312, zShared_dim * 2)
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
        # hiddens = self.enc_hidden(attributes)
        stats = self.fc(attributes)

        muShared = stats[:, :, :self.zShared_dim]
        logvarShared = stats[:, :, self.zShared_dim:]
        stdShared = torch.sqrt(torch.exp(logvarShared) + EPS)
        # attributes
        q.normal(loc=muShared,
                 scale=stdShared,
                 name='sharedB')
        return q


class DecoderB(nn.Module):
    def __init__(self, seed, zShared_dim, num_hidden):
        super(self.__class__, self).__init__()
        self.seed = seed
        self.zShared_dim = zShared_dim

        # self.dec_hidden = nn.Sequential(
        #     nn.Linear(zShared_dim, num_hidden),
        #     nn.ReLU(),
        # )
        self.dec_label = nn.Sequential(
            nn.Linear(zShared_dim, 312))
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
        shared_mean = torch.zeros_like(q['sharedB'].dist.loc)
        shared_std = torch.ones_like(q['sharedB'].dist.scale)
        for shared_from in shared.keys():
            # prior for z_shared_atrr
            zShared = p.normal(shared_mean,
                               shared_std,
                               value=q[shared[shared_from]],
                               name=shared[shared_from])

            # hiddens = self.dec_hidden(zShared)
            pred_labels = self.dec_label(zShared)
            pred_labels = pred_labels.squeeze(0)

            pred_labels = F.logsigmoid(pred_labels + EPS)

            p.loss(
                lambda y_pred, target: F.binary_cross_entropy_with_logits(y_pred, target, reduction='none').sum(
                    dim=1) / N_ATTR, \
                pred_labels, attributes, name='attr_' + shared_from)
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
