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
    def __init__(self, num_pixels=784,
                       num_hidden=256,
                       num_digits=10,
                       num_style=50):
        super(self.__class__, self).__init__()
        self.enc_hidden = nn.Sequential(
            nn.Linear(num_pixels, num_hidden),
            nn.ReLU())
        self.digit_log_weights = nn.Linear(num_hidden, num_digits)
        self.digit_temp = torch.tensor(0.66)
        self.style_mean = nn.Linear(num_hidden + num_digits, num_style)
        self.style_log_std = nn.Linear(num_hidden + num_digits, num_style)

    @expand_inputs
    def forward(self, images, labels=None, num_samples=None):
        q = probtorch.Trace()
        hiddens = self.enc_hidden(images)
        digits = q.concrete(logits=self.digit_log_weights(hiddens),
                            temperature=self.digit_temp,
                            value=labels,
                            name='digits')
        hiddens2 = torch.cat([digits, hiddens], -1)
        styles_mean = self.style_mean(hiddens2)
        styles_std = torch.exp(self.style_log_std(hiddens2))
        q.normal(styles_mean,
                 styles_std,
                 name='styles')
        return q

class Decoder(nn.Module):
    def __init__(self, num_pixels=784,
                       num_hidden=256,
                       num_digits=10,
                       num_style=50):
        super(self.__class__, self).__init__()
        self.num_digits = num_digits
        self.digit_log_weights = torch.zeros(num_digits)
        self.digit_temp = 0.66
        self.style_mean = torch.zeros(num_style)
        self.style_std = torch.ones(num_style)
        self.dec_hidden = nn.Sequential(
                            nn.Linear(num_style + num_digits, num_hidden),
                            nn.ReLU())
        self.dec_image = nn.Sequential(
                           nn.Linear(num_hidden, num_pixels),
                           nn.Sigmoid())

    def forward(self, images, q=None, num_samples=None):
        p = probtorch.Trace()
        digits = p.concrete(logits=self.digit_log_weights,
                            temperature=self.digit_temp,
                            value=q['digits'],
                            name='digits')
        styles = p.normal(self.style_mean,
                          self.style_std,
                          value=q['styles'],
                          name='styles')
        hiddens = self.dec_hidden(torch.cat([digits, styles], -1))
        images_mean = self.dec_image(hiddens)
        p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                  torch.log(1 - x_hat + EPS) * (1-x)).sum(-1),
               images_mean, images, name='images')
        return p