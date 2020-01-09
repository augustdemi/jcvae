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
                 zShared_dim=18):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zShared_dim = zShared_dim
        self.seed = seed

        self.enc_hidden = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.Conv2d(128, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            Swish())

        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 512),
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(512, zShared_dim * 2))

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

        hiddens = self.enc_hidden(x)
        hiddens = hiddens.view(hiddens.size(0), -1)
        stats = self.fc(hiddens)
        stats = stats.unsqueeze(0)

        muShared = stats[:, :, :self.zShared_dim]
        logvarShared = stats[:, :, self.zShared_dim:]
        stdShared = torch.sqrt(torch.exp(logvarShared) + EPS)

        # attributes
        try:
            q.normal(loc=muShared,
                     scale=stdShared,
                     name='sharedA')
        except:
            print('muShared', muShared)
            print('stdShared', stdShared)
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
            Swish()
        )

        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
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
        shared_mean = torch.zeros_like(q['sharedA'].dist.loc)
        shared_std = torch.ones_like(q['sharedA'].dist.scale)

        p = probtorch.Trace()

        for shared_from in shared.keys():
            # prior for z_shared_atrr
            zShared = p.normal(shared_mean,
                               shared_std,
                               value=q[shared[shared_from]],
                               name=shared[shared_from])
            hiddens = self.fc(zShared)
            hiddens = hiddens.view(-1, 256, 5, 5)
            images_mean = self.hallucinate(hiddens)
            images_mean = images_mean.view(images_mean.size(0), -1)
            images = images.view(images.size(0), -1)
            # define reconstruction loss (log prob of bernoulli dist)
            p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                      torch.log(1 - x_hat + EPS) * (1 - x)).sum(-1),
                   images_mean, images, name='images_' + shared_from)
        return p

    def forward2(self, zShared, cuda):
        hiddens = self.fc(zShared)
        hiddens = hiddens.view(-1, 256, 5, 5)
        images_mean = self.hallucinate(hiddens)
        return images_mean


class EncoderB(nn.Module):
    def __init__(self, seed, num_attr=18,
                 num_hidden=512,
                 zShared_dim=18):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zShared_dim = zShared_dim
        self.seed = seed
        self.enc_hidden = nn.Sequential(
            nn.Linear(num_attr, num_hidden),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(num_hidden, num_hidden),
            nn.BatchNorm1d(512),
            Swish(),
        )

        self.fc = nn.Linear(num_hidden, 2 * zShared_dim)
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    # @expand_inputs
    def forward(self, attributes, cuda, num_samples=None, q=None):
        if q is None:
            q = probtorch.Trace()
        hiddens = self.enc_hidden(attributes)
        stats = self.fc(hiddens)

        muShared = stats[:, :self.zShared_dim].unsqueeze(0)
        logvarShared = stats[:, self.zShared_dim:].unsqueeze(0)
        stdShared = torch.sqrt(torch.exp(logvarShared) + EPS)

        # attributes
        q.normal(loc=muShared,
                 scale=stdShared,
                 name='sharedB')
        return q


class DecoderB(nn.Module):
    def __init__(self, seed, num_attr=18,
                 num_hidden=512,
                 zShared_dim=18):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP
        self.num_attr = num_attr
        self.seed = seed

        self.dec_hidden = nn.Sequential(
            nn.Linear(zShared_dim, num_hidden),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(num_hidden, num_hidden),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Linear(num_hidden, num_hidden),
            nn.BatchNorm1d(512),
            Swish(),
        )
        self.dec_label = nn.Sequential(
            nn.Linear(num_hidden, num_attr))
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    def forward(self, attributes, shared, q=None, p=None, num_samples=None, train=True, CUDA=False):
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
            hiddens = self.dec_hidden(zShared.squeeze(0))
            pred_labels = self.dec_label(hiddens)

            pred_labels = F.logsigmoid(pred_labels + EPS)

            p.loss(
                lambda y_pred, target: F.binary_cross_entropy_with_logits(y_pred, target, reduction='none').sum(dim=1), \
                pred_labels, attributes, name='attr_' + shared_from)
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
