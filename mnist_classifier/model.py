import numpy as np
import torch
import torch.nn as nn

import sys

sys.path.append('../')
import probtorch
from probtorch.util import expand_inputs, normal_init, kaiming_init
from torch.nn import functional as F

EPS = 1e-9
TEMP = 0.66


class EncoderA(nn.Module):
    def __init__(self, seed):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.seed = seed

        self.enc_hidden = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU())

        self.fc = nn.Linear(256, 10)
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    def forward(self, x, num_samples=None, q=None):
        hiddens = self.enc_hidden(x)
        hiddens = hiddens.view(hiddens.size(0), -1)
        prediction = self.fc(hiddens)
        prediction = F.log_softmax(prediction + EPS, dim=1)
        return prediction
