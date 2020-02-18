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


class EncoderA(nn.Module):
    def __init__(self, seed):
        super(self.__class__, self).__init__()
        self.seed = seed
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]  # we do not use the last fc layer.
        self.resnet = nn.Sequential(*modules)
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
        feature = self.resnet(x)
        feature = feature.view(feature.size(0), -1)
        return feature


class DecoderA2(nn.Module):
    def __init__(self, seed):
        super(self.__class__, self).__init__()
        self.seed = seed

        self.dec_image = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid())

        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    def forward(self, features):
        x = features.view(features.size(0), 512, 2, 2)
        x = self.dec_image(x)
        return x
