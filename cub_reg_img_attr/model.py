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
    def __init__(self, seed, zPrivate_dim, zSharedAttr_dim):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zPrivate_dim = zPrivate_dim
        self.zSharedAttr_dim = zSharedAttr_dim
        self.seed = seed

        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # we do not use the last fc layer.
        self.resnet = nn.Sequential(*modules)

        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(2048, sum(zSharedAttr_dim)),
            nn.Tanh()
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
    def forward(self, x, attributes, num_samples=None, q=None):
        if q is None:
            q = probtorch.Trace()
        hiddens = self.resnet(x)
        hiddens = hiddens.view(hiddens.size(0), -1)
        pred_labels = self.fc(hiddens)
        # pred_labels = stats.unsqueeze(0)
        # pred_labels = pred_labels.squeeze(0)

        pred_labels = F.logsigmoid(pred_labels + EPS)
        loss = F.binary_cross_entropy_with_logits(pred_labels, attributes, reduction='none').sum()
        acc = (torch.round(torch.exp(pred_labels)) == attributes).sum() / sum(self.zSharedAttr_dim)
        return loss, acc
