import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append('../')
import probtorch
from probtorch.util import expand_inputs, normal_init
from torch.nn import functional as F

EPS = 1e-9
TEMP = 0.66

class EncoderA(nn.Module):
    def __init__(self, num_pixels=784,
                       num_hidden=256,
                       zShared_dim=10,
                     zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zPrivate_dim = zPrivate_dim
        self.zShared_dim = zShared_dim

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)
        self.fc5 = nn.Linear(64 * 4 * 4, 256)
        self.fc6 = nn.Linear(256, 2 * zPrivate_dim + 2 * zShared_dim)

        # initialize parameters
        self.weight_init()

    ####
    def weight_init(self):
        initializer = normal_init
        for m in self._modules:
            initializer(self._modules[m])

    # @expand_inputs
    def forward(self, x, num_samples=None, q=None):
        # print(self._modules['enc_hidden'][0].weight.sum())
        if q is None:
            q = probtorch.Trace()

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc5(out))
        stats = self.fc6(out)
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
        q.normal(loc=muShared,
                 scale=stdShared,
                 name='sharedA')
        return q


class DecoderA(nn.Module):
    def __init__(self, num_pixels=784,
                       num_hidden=256,
                    zShared_dim=10,
                    zPrivate_dim=50):
        super(self.__class__, self).__init__()

        self.fc1 = nn.Linear(zPrivate_dim + zShared_dim, 256)
        self.fc2 = nn.Linear(256, 4 * 4 * 64)
        self.deconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.deconv6 = nn.ConvTranspose2d(32, 1, 4, 2, 1)
        # initialize parameters
        self.weight_init()

    ####
    def weight_init(self):
        initializer = normal_init
        for m in self._modules:
            initializer(self._modules[m])

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
        # private은 sharedA(infA), sharedB(crossA), sharedPOE 모두에게 공통적으로 들어가는 node로 z_private 한 샘플에 의해 모두가 다 생성돼야함
        for shared_name in shared.keys():
            # prior for z_shared
            zShared = p.normal(shared_mean,
                               shared_std,
                                value=shared[shared_name],
                                name=shared_name)

            z = torch.cat([zPrivate, zShared], -1)
            out = F.relu(self.fc1(z))
            out = F.relu(self.fc2(out))
            out = out.view(-1, 64, 4, 4)
            out = F.relu(self.deconv3(out))
            out = F.relu(self.deconv4(out))
            out = F.relu(self.deconv5(out))
            images_mean = F.sigmoid(self.deconv6(out))

            images_mean = images_mean.view(images_mean.size(0), -1)
            images = images.view(images.size(0), -1)
            # define reconstruction loss (log prob of bernoulli dist)
            p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                      torch.log(1 - x_hat + EPS) * (1-x)).sum(-1),
                   images_mean, images, name= 'imagesA_' + shared_name)
        return p

    def forward2(self, zPrivate, zShared):
        z = torch.cat([zPrivate, zShared], -1)
        out = F.relu(self.fc1(z))
        out = F.relu(self.fc2(out))
        out = out.view(-1, 64, 4, 4)
        out = F.relu(self.deconv3(out))
        out = F.relu(self.deconv4(out))
        out = F.relu(self.deconv5(out))
        images_mean = F.sigmoid(self.deconv6(out))
        return images_mean

class EncoderB(nn.Module):
    def __init__(self, num_pixels=784,
                       num_hidden=256,
                       zShared_dim=10,
                     zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zPrivate_dim = zPrivate_dim
        self.zShared_dim = zShared_dim

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)
        self.fc5 = nn.Linear(64 * 4 * 4, 256)
        self.fc6 = nn.Linear(256, 2 * zPrivate_dim + 2 * zShared_dim)

        # initialize parameters
        self.weight_init()

    ####
    def weight_init(self):
        initializer = normal_init
        for m in self._modules:
            initializer(self._modules[m])

    # @expand_inputs
    def forward(self, x, num_samples=None, q=None):
        if q is None:
            q = probtorch.Trace()

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc5(out))
        stats = self.fc6(out)
        stats = stats.unsqueeze(0)

        muPrivate = stats[:, :, :self.zPrivate_dim]
        logvarPrivate = stats[:, :, self.zPrivate_dim:(2 * self.zPrivate_dim)]
        stdPrivate = torch.sqrt(torch.exp(logvarPrivate) + EPS)


        muShared = stats[:, :, (2 * self.zPrivate_dim):(2 * self.zPrivate_dim + self.zShared_dim)]
        logvarShared = stats[:, :, (2 * self.zPrivate_dim + self.zShared_dim):]
        stdShared = torch.sqrt(torch.exp(logvarShared) + EPS)

        q.normal(loc=muPrivate,
                 scale=stdPrivate,
                 name='privateB')
        q.normal(loc=muShared,
                 scale=stdShared,
                 name='sharedB')
        return q


class DecoderB(nn.Module):
    def __init__(self, num_pixels=784,
                       num_hidden=256,
                    zShared_dim=10,
                    zPrivate_dim=50):
        super(self.__class__, self).__init__()
        self.fc1 = nn.Linear(zPrivate_dim + zShared_dim, 256)
        self.fc2 = nn.Linear(256, 4 * 4 * 64)
        self.deconv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.deconv6 = nn.ConvTranspose2d(32, 1, 4, 2, 1)
        # initialize parameters
        self.weight_init()

    ####
    def weight_init(self):
        initializer = normal_init
        for m in self._modules:
            initializer(self._modules[m])

    def forward(self, images, shared, q=None, p=None, num_samples=None):
        priv_mean = torch.zeros_like(q['privateB'].dist.loc)
        priv_std = torch.ones_like(q['privateB'].dist.scale)

        shared_mean = torch.zeros_like(q['sharedB'].dist.loc)
        shared_std = torch.ones_like(q['sharedB'].dist.scale)

        p = probtorch.Trace()

        # prior for z_private
        zPrivate = p.normal(priv_mean,
                            priv_std,
                        value=q['privateB'],
                        name='privateB')
        # private은 sharedA(infA), sharedB(crossA), sharedPOE 모두에게 공통적으로 들어가는 node로 z_private 한 샘플에 의해 모두가 다 생성돼야함
        for shared_name in shared.keys():
            # prior for z_shared
            zShared = p.normal(shared_mean,
                               shared_std,
                                value=shared[shared_name],
                                name=shared_name)

            z = torch.cat([zPrivate, zShared], -1)
            out = F.relu(self.fc1(z))
            out = F.relu(self.fc2(out))
            out = out.view(-1, 64, 4, 4)
            out = F.relu(self.deconv3(out))
            out = F.relu(self.deconv4(out))
            out = F.relu(self.deconv5(out))
            images_mean = F.sigmoid(self.deconv6(out))

            images_mean = images_mean.view(images_mean.size(0), -1)
            images = images.view(images.size(0), -1)

            # define reconstruction loss (log prob of bernoulli dist)
            p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                      torch.log(1 - x_hat + EPS) * (1-x)).sum(-1),
                   images_mean, images, name= 'imagesB_' + shared_name)
        return p

    def forward2(self, zPrivate, zShared):
        z = torch.cat([zPrivate, zShared], -1)
        out = F.relu(self.fc1(z))
        out = F.relu(self.fc2(out))
        out = out.view(-1, 64, 4, 4)
        out = F.relu(self.deconv3(out))
        out = F.relu(self.deconv4(out))
        out = F.relu(self.deconv5(out))
        images_mean = F.sigmoid(self.deconv6(out))
        return images_mean
