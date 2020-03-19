import torch
import torch.nn as nn
import sys

sys.path.append('../')
import probtorch
from probtorch.util import expand_inputs, normal_init, kaiming_init
from torch.nn import functional as F

EPS = 1e-9
TEMP = 0.66


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""

    def forward(self, x):
        return x * F.sigmoid(x)

class EncoderA(nn.Module):
    def __init__(self, seed,
                 zShared_dim=64):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zShared_dim = zShared_dim
        self.seed = seed

        # self.enc_hidden = nn.Sequential(
        #     nn.Linear(784, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU())

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, zShared_dim)
        self.fc32 = nn.Linear(512, zShared_dim)
        self.swish = Swish()
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    def forward(self, x, cuda, num_samples=None, q=None):
        if q is None:
            q = probtorch.Trace()

        # h = self.enc_hidden(x.view(-1, 784))
        h = F.relu(self.fc1(x.view(-1, 784)))
        h = F.relu(self.fc2(h))
        muShared = self.fc31(h).unsqueeze(0)
        logvarShared = self.fc32(h).unsqueeze(0)
        stdShared = torch.sqrt(torch.exp(logvarShared) + EPS)

        q.normal(loc=muShared,
                 scale=stdShared,
                 name='sharedA')
        return q


class DecoderA(nn.Module):
    def __init__(self, seed,
                 zShared_dim=18):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP
        self.num_digits = zShared_dim
        self.seed = seed

        # self.dec_hidden = nn.Sequential(
        #     nn.Linear(zShared_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU()
        # )
        self.fc1 = nn.Linear(zShared_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.dec_image = nn.Sequential(
            nn.Linear(512, 784),
            nn.Sigmoid())

        self.swish = Swish()
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    def forward(self, images, shared, q=None, p=None, num_samples=None):
        shared_mean = torch.zeros_like(q['sharedA'].dist.loc)
        shared_std = torch.ones_like(q['sharedA'].dist.scale)

        p = probtorch.Trace()

        for shared_from in shared.keys():
            # prior for z_shared_atrr
            zShared = p.normal(shared_mean,
                               shared_std,
                               value=q[shared[shared_from]],
                               name=shared[shared_from])

            # h = self.dec_hidden(zShared.squeeze(0))
            h = F.relu(self.fc1(zShared.squeeze(0)))
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
            images_mean = self.dec_image(h)
            # define reconstruction loss (log prob of bernoulli dist)
            p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                      torch.log(1 - x_hat + EPS) * (1 - x)).sum(-1),
                   images_mean, images, name='images_' + shared_from)
        return p

    def forward2(self, zShared, cuda):
        h = self.dec_hidden(zShared.squeeze(0))
        images_mean = self.dec_image(h)
        return images_mean


class EncoderB(nn.Module):
    def __init__(self, seed,
                 zShared_dim=10):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zShared_dim = zShared_dim
        self.seed = seed

        # self.enc_hidden = nn.Sequential(
        #     nn.Embedding(10, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU()
        # )

        self.fc1 = nn.Embedding(10, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, zShared_dim)
        self.fc32 = nn.Linear(512, zShared_dim)
        self.swish = Swish()

        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    def forward(self, labels, cuda, num_samples=None, q=None):
        if q is None:
            q = probtorch.Trace()
        # h = self.enc_hidden(labels)
        h = F.relu(self.fc1(labels))
        h = F.relu(self.fc2(h))
        muShared = self.fc31(h).unsqueeze(0)
        logvarShared = self.fc32(h).unsqueeze(0)
        stdShared = torch.sqrt(torch.exp(logvarShared) + EPS)

        q.normal(loc=muShared,
                 scale=stdShared,
                 name='sharedB')
        return q


class DecoderB(nn.Module):
    def __init__(self, seed,
                 zShared_dim=18):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP
        self.seed = seed

        # self.dec_hidden = nn.Sequential(
        #     nn.Linear(zShared_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU()
        # )

        self.fc1 = nn.Linear(zShared_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 10)

        self.swish = Swish()
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    def forward(self, labels, shared, q=None, p=None, num_samples=None, train=True, CUDA=False):
        shared_mean = torch.zeros_like(q['sharedB'].dist.loc)
        shared_std = torch.ones_like(q['sharedB'].dist.scale)
        pred = {}

        p = probtorch.Trace()

        for shared_from in shared.keys():
            # prior for z_shared_atrr
            zShared = p.normal(shared_mean,
                               shared_std,
                               value=q[shared[shared_from]],
                               name=shared[shared_from])

            # h = self.dec_hidden(zShared.squeeze(0))
            h = F.relu(self.fc1(zShared.squeeze(0)))
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
            pred_labels = self.fc4(h)

            pred_labels = F.log_softmax(pred_labels + EPS, dim=1)

            p.loss(lambda y_pred, target: -(target * y_pred).sum(-1), \
                   pred_labels, labels, name='label_' + shared_from)
            pred.update({shared_from: pred_labels})

        if train:
            predicted_attr = pred['own']
        else:
            predicted_attr = pred['cross']
        return p, predicted_attr

