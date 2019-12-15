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


class linearRegression(nn.Module):
    def __init__(self, seed, zSharedAttr_dim, zSharedLabel_dim):
        super(self.__class__, self).__init__()
        self.seed = seed
        self.fc = nn.Linear(sum(zSharedAttr_dim), zSharedLabel_dim)
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    def forward(self, attributes, labels):
        pred_labels = self.fc(attributes)
        pred_labels = F.log_softmax(pred_labels + EPS, dim=1)
        loss = (pred_labels * labels).sum()
        acc = (pred_labels.max(-1)[1] == labels.max(-1)[1]).float().sum()
        return loss, acc
