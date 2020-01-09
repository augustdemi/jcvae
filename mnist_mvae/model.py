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
    def __init__(self, seed,
                 zShared_dim=64):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zShared_dim = zShared_dim
        self.seed = seed

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

        h = self.swish(self.fc1(x.view(-1, 784)))
        h = self.swish(self.fc2(h))

        muShared = self.fc31(h).unsqueeze(0)
        logvarShared = self.fc32(h).unsqueeze(0)
        stdShared = torch.sqrt(torch.exp(logvarShared) + EPS)

        mu_poe, std_poe = probtorch.util.apply_poe(cuda, muShared, stdShared)

        q.normal(loc=mu_poe,
                 scale=std_poe,
                 name='sharedA')
        return q


class DecoderA(nn.Module):
    def __init__(self, seed,
                 zShared_dim=18):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP
        self.num_digits = zShared_dim
        self.seed = seed

        self.fc = nn.Sequential(
            nn.Linear(zShared_dim, 256 * 5 * 5),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(zShared_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 784)
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

            h = self.swish(self.fc1(zShared.squeeze(0)))
            h = self.swish(self.fc2(h))
            h = self.swish(self.fc3(h))
            images_mean = self.fc4(h)
            images_mean = F.sigmoid(images_mean)
            images = images.view(images.size(0), -1)
            # define reconstruction loss (log prob of bernoulli dist)
            p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                      torch.log(1 - x_hat + EPS) * (1 - x)).sum(-1),
                   images_mean, images, name='images_' + shared_from)
        return p

    def forward2(self, zShared, cuda):
        h = self.swish(self.fc1(zShared.squeeze(0)))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
        images_mean = self.fc4(h)
        images_mean = F.sigmoid(images_mean)
        return images_mean


class EncoderB(nn.Module):
    def __init__(self, seed,
                 zShared_dim=10):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zShared_dim = zShared_dim
        self.seed = seed

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
        h = self.swish(self.fc1(labels))
        h = self.swish(self.fc2(h))

        muShared = self.fc31(h).unsqueeze(0)
        logvarShared = self.fc32(h).unsqueeze(0)
        stdShared = torch.sqrt(torch.exp(logvarShared) + EPS)

        mu_poe, std_poe = probtorch.util.apply_poe(cuda, muShared, stdShared)
        # attributes
        q.normal(loc=mu_poe,
                 scale=std_poe,
                 name='sharedB')
        return q


class DecoderB(nn.Module):
    def __init__(self, seed,
                 zShared_dim=18):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP
        self.seed = seed

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

            h = self.swish(self.fc1(zShared.squeeze(0)))
            h = self.swish(self.fc2(h))
            h = self.swish(self.fc3(h))
            pred_labels = self.fc4(h)

            pred_labels = F.log_softmax(pred_labels + EPS, dim=1)

            p.loss(lambda y_pred, target: -(target * y_pred).sum(-1), \
                   pred_labels, labels.unsqueeze(0), name='label_' + shared_from)
            pred.update({shared_from: pred_labels})

        if train:
            predicted_attr = pred['own']
        else:
            predicted_attr = pred['cross']
        return p, predicted_attr


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""

    def forward(self, x):
        return x * F.sigmoid(x)