import torch
import torch.nn as nn

import sys

sys.path.append('../')
import probtorch
from probtorch.util import expand_inputs, kaiming_init
from torch.nn import functional as F

EPS = 1e-9
TEMP = 0.66


class EncoderA(nn.Module):
    def __init__(self, seed,
                 zShared_dim=10):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zShared_dim = zShared_dim
        self.seed = seed

        self.enc_hidden = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2 * zShared_dim))
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    def forward(self, x, num_samples=None, q=None):
        if q is None:
            q = probtorch.Trace()

        hiddens = self.enc_hidden(x)
        hiddens = hiddens.view(hiddens.size(0), -1)
        stats = self.fc(hiddens)
        stats = stats.unsqueeze(0)

        muShared = stats[:, :, :self.zShared_dim]
        logvarShared = stats[:, :, self.zShared_dim:]
        stdShared = torch.sqrt(torch.exp(logvarShared) + EPS)


        q.normal(loc=muShared,
                 scale=stdShared,
                 name='sharedA')
        return q


class DecoderA(nn.Module):
    def __init__(self, seed,
                 zShared_dim=10):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP

        self.seed = seed

        self.dec_hidden = nn.Sequential(
            nn.Linear(zShared_dim, 256 * 2 * 2),
            nn.ReLU())
        self.dec_image = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.ReLU(),
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

        # private은 sharedA(infA), sharedB(crossA), sharedPOE 모두에게 공통적으로 들어가는 node로 z_private 한 샘플에 의해 모두가 다 생성돼야함
        for shared_name in shared.keys():
            # prior for z_shared
            zShared = p.normal(shared_mean,
                               shared_std,
                               value=shared[shared_name],
                               name=shared_name)

            hiddens = self.dec_hidden(zShared)
            hiddens = hiddens.view(-1, 256, 2, 2)
            images_mean = self.dec_image(hiddens)

            images_mean = images_mean.view(images_mean.size(0), -1)
            images = images.view(images.size(0), -1)
            # define reconstruction loss (log prob of bernoulli dist)
            p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
                                      torch.log(1 - x_hat + EPS) * (1 - x)).sum(-1),
                   images_mean, images, name='images_' + shared_name)
        return p

    def make_one_hot(self, alpha, cuda):
        _, max_alpha = torch.max(alpha, dim=1)
        one_hot_samples = torch.zeros(alpha.size())
        one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
        if cuda:
            one_hot_samples = one_hot_samples.cuda()
        return one_hot_samples

    def forward2(self, zPrivate, zShared, cuda):
        zShared = self.make_one_hot(zShared.squeeze(0), cuda).unsqueeze(0)
        hiddens = self.dec_hidden(torch.cat([zPrivate, zShared], -1))
        images_mean = self.dec_image(hiddens)
        return images_mean

    def loglike(self, images, shared, q=None, p=None, num_samples=None):
        digit_log_weights = torch.zeros_like(
            q['sharedA'].dist.logits)  # prior is the concrete dist for uniform dist. with all params=1
        style_mean = torch.zeros_like(q['privateA'].dist.loc)
        style_std = torch.ones_like(q['privateA'].dist.scale)

        p = probtorch.Trace()

        # prior for z_private
        zPrivate = p.normal(style_mean,
                            style_std,
                            value=q['privateA'],
                            name='privateA')
        # private은 sharedA(infA), sharedB(crossA), sharedPOE 모두에게 공통적으로 들어가는 node로 z_private 한 샘플에 의해 모두가 다 생성돼야함

        importance_ll = {}
        for shared_name in shared.keys():
            # prior for z_shared
            zShared = p.concrete(logits=digit_log_weights,
                                 temperature=self.digit_temp,
                                 value=shared[shared_name],
                                 name=shared_name)

            if 'poe' in shared_name:
                hiddens = self.dec_hidden(torch.cat([zPrivate, torch.pow(zShared + EPS, 1 / 3)], -1))
            else:
                hiddens = self.dec_hidden(torch.cat([zPrivate, zShared], -1))

            images_mean = self.dec_image(hiddens)

            # define reconstruction loss (log prob of bernoulli dist)
            log_p_x_given_z = (
                torch.log(images_mean + EPS) * images + torch.log(1 - images_mean + EPS) * (1 - images)).sum(-1)
            # sampling from q(z|x)
            log_q_z_given_x = shared[shared_name].log_prob
            # sampling from p(z)
            p.concrete(logits=digit_log_weights,
                       temperature=self.digit_temp, name=shared_name + '_prior')
            log_p_z = p[shared_name + '_prior'].log_prob

            importance_ll.update({shared_name: log_p_x_given_z + log_p_z - log_q_z_given_x})
            # importance_ll.update({shared_name: log_p_x_given_z})

        return importance_ll


class EncoderB(nn.Module):
    def __init__(self, seed, num_digis=10,
                 num_hidden=256,
                 zShared_dim=10):
        super(self.__class__, self).__init__()
        self.digit_temp = torch.tensor(TEMP)
        self.zShared_dim = zShared_dim
        self.seed = seed
        self.enc_hidden = nn.Sequential(
            nn.Linear(num_digis, num_hidden),
            nn.ReLU())

        self.fc = nn.Linear(num_hidden, 2 * zShared_dim)
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    @expand_inputs
    def forward(self, labels, num_samples=None, q=None):
        if q is None:
            q = probtorch.Trace()
        hiddens = self.enc_hidden(labels)
        stats = self.fc(hiddens)

        muShared = stats[:, :, :self.zShared_dim]
        logvarShared = stats[:, :, self.zShared_dim:]
        stdShared = torch.sqrt(torch.exp(logvarShared) + EPS)

        # attributes
        q.normal(loc=muShared,
                 scale=stdShared,
                 name='sharedB')
        return q


class DecoderB(nn.Module):
    def __init__(self, seed, num_digits=10,
                 num_hidden=256,
                 zShared_dim=10):
        super(self.__class__, self).__init__()
        self.digit_temp = TEMP
        self.seed = seed

        self.dec_hidden = nn.Sequential(
            nn.Linear(zShared_dim, num_hidden),
            nn.ReLU())
        self.dec_label = nn.Sequential(
            nn.Linear(num_hidden, num_digits))
        self.weight_init()

    def weight_init(self):
        for m in self._modules:
            if isinstance(self._modules[m], nn.Sequential):
                for one_module in self._modules[m]:
                    kaiming_init(one_module, self.seed)
            else:
                kaiming_init(self._modules[m], self.seed)

    def forward(self, labels, shared, q=None, p=None, num_samples=None, train=True):
        shared_mean = torch.zeros_like(q['sharedB'].dist.loc)
        shared_std = torch.ones_like(q['sharedB'].dist.scale)

        p = probtorch.Trace()
        # private은 sharedA(infA), sharedB(crossA), sharedPOE 모두에게 공통적으로 들어가는 node로 z_private 한 샘플에 의해 모두가 다 생성돼야함
        for shared_name in shared.keys():
            # prior for z_shared # prior is the concrete dist for uniform dist. with all params=1
            zShared = p.normal(shared_mean,
                               shared_std,
                               value=shared[shared_name],
                               name=shared_name)
            hiddens = self.dec_hidden(zShared)

            pred_labels = self.dec_label(hiddens)
            # define reconstruction loss (log prob of bernoulli dist)
            pred_labels = F.log_softmax(pred_labels + EPS, dim=2)
            if train:
                p.loss(lambda y_pred, target: -(target * y_pred).sum(-1), \
                       pred_labels, labels.unsqueeze(0), name='labels_' + shared_name)
            else:
                p.loss(lambda y_pred, target: (1 - (target == y_pred).float()), \
                       pred_labels.max(-1)[1], labels.max(-1)[1], name='labels_' + shared_name)

        return p
