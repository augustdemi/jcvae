import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from torch.nn import Parameter
import torch
import torch.nn as nn

import sys

sys.path.append('../')
import probtorch
from probtorch.util import expand_inputs

# model parameters
# NUM_PIXELS = 784
# NUM_HIDDEN = 256
# NUM_DIGITS = 10
# NUM_STYLE = 50
#
# # training parameters
# NUM_SAMPLES = 8
# NUM_BATCH = 128
# NUM_EPOCHS = 200
# LABEL_FRACTION = 0.01
# LEARNING_RATE = 1e-3
# BETA1 = 0.90
EPS = 1e-9


# #CUDA = torch.cuda.is_available()
# CUDA = True
#
# # path parameters
# MODEL_NAME = 'mnist-semisupervised-%02ddim' % NUM_STYLE
# DATA_PATH = '../data'
# WEIGHTS_PATH = '../weights'
# RESTORE = False




class Encoder(nn.Module):
    def __init__(self,
                 num_attr=18,
                 num_style=50):
        super(self.__class__, self).__init__()
        self.num_attr = num_attr
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

        num_hidden = 512
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, num_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1))

        self.attr_log_weights = nn.Linear(num_hidden, num_attr * 2)
        self.digit_temp = torch.tensor(0.66)
        self.style_mean = nn.Linear(num_hidden + num_attr * 2, num_style)
        self.style_log_std = nn.Linear(num_hidden + num_attr * 2, num_style)

    @expand_inputs
    def forward(self, images, attr=None, num_samples=None):
        q = probtorch.Trace()
        images = images.squeeze(0)
        hiddens = self.enc_hidden(images)
        hiddens = hiddens.view(hiddens.size(0), -1)
        hiddens = self.fc(hiddens)

        attr_log_weights = self.attr_log_weights(hiddens).unsqueeze(0)
        # attributes

        attrs = []
        labels_onehot = None
        for i in range(self.num_attr):
            if attr:
                labels_onehot = torch.zeros((100, 2))
                labels_onehot.scatter_(1, attr[:, :, i].squeeze(0).unsqueeze(1).type(torch.long), 1)
                labels_onehot = torch.clamp(labels_onehot, EPS, 1 - EPS).unsqueeze(0)

            attrs.append(q.concrete(logits=attr_log_weights[:, :, i * 2:(i + 1) * 2],
                                    temperature=self.digit_temp,
                                    value=labels_onehot,
                                    name='attr' + str(i)))
        attrs.append(hiddens.unsqueeze(0))

        hiddens2 = torch.cat(attrs, -1)
        styles_mean = self.style_mean(hiddens2)
        styles_std = torch.exp(self.style_log_std(hiddens2))
        q.normal(styles_mean,
                 styles_std,
                 name='styles')
        return q


class Decoder(nn.Module):
    def __init__(self,
                 num_attr=18,
                 num_style=50):
        super(self.__class__, self).__init__()
        self.num_attr = num_attr
        self.digit_temp = 0.66
        self.attr_log_weights = torch.zeros(2)
        self.style_mean = torch.zeros(num_style)
        self.style_std = torch.ones(num_style)

        num_hidden = 256 * 5 * 5
        self.dec_hidden = nn.Sequential(
            nn.Linear(num_style + 2 * num_attr, num_hidden),
            nn.ReLU())
        self.dec_image = nn.Sequential(
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

    def forward(self, images, q=None, num_samples=None):
        p = probtorch.Trace()

        attrs = []

        for i in range(self.num_attr):
            attrs.append(p.concrete(logits=self.attr_log_weights,
                                    temperature=self.digit_temp,
                                    value=q['attr' + str(i)],
                                    name='attr' + str(i)))

        styles = p.normal(self.style_mean,
                          self.style_std,
                          value=q['styles'],
                          name='styles')
        attrs.append(styles)

        hiddens = self.dec_hidden(torch.cat(attrs, -1))
        hiddens = hiddens.view(-1, 256, 5, 5)
        images_mean = self.dec_image(hiddens)

        images_mean = images_mean.view(images_mean.size(0), -1).unsqueeze(0)
        images = images.view(images.size(0), -1).unsqueeze(0)
        p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                  torch.log(1 - x_hat + EPS) * (1 - x)).sum(-1),
               images_mean, images, name='images')
        return p
