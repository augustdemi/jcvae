"Helper functions that don't have a better place yet"
import torch
from numbers import Number
import math
from functools import wraps
import torch.nn as nn
import torch.nn.init as init
import os
import imageio
import subprocess
import numpy as np

from torchvision import transforms


__all__ = ['broadcast_size',
           'expanded_size',
           'batch_sum',
           'partial_sum',
           'log_sum_exp',
           'log_mean_exp']


def expand_inputs(f):
    """Decorator that expands all input tensors to add a sample dimensions"""
    @wraps(f)
    def g(*args, **kwargs):
        num_samples = kwargs.get('num_samples', None)
        if num_samples is not None:
            new_args = []
            new_kwargs = {}
            for arg in args:
                if hasattr(arg, 'expand'):
                    new_args.append(arg.expand(num_samples, *arg.size()))
                else:
                    new_args.append(arg)
            for k in kwargs:
                arg = kwargs[k]
                if hasattr(arg, 'expand'):
                    new_kwargs[k] = arg.expand(num_samples, *arg.size())
                else:
                    new_kwargs[k] = arg
                new_kwargs['num_samples'] = num_samples
            return f(*new_args, **new_kwargs)
        else:
            return f(*args, **kwargs)
    return g


def broadcast_size(a, b):
    """Returns the broadcasted size given two Tensors or Variables"""
    a_size = torch.Size((1,)) if isinstance(a, Number) else a.size()
    b_size = torch.Size((1,)) if isinstance(b, Number) else b.size()
    # order a and b by number of dimensions
    if len(b_size) > len(a_size):
        a_size, b_size = b_size, a_size
    # pad b with 1's if needed
    b_size = torch.Size((1,) * (len(a_size) - len(b_size))) + b_size
    c_size = a_size[0:0]
    for a, b in zip(a_size, b_size):
        if a == 1:
            c_size += (b,)
        elif b == 1:
            c_size += (a,)
        else:
            if a != b:
                raise ValueError("Broadcasting dimensions must be either equal"
                                 "or 1.")
            c_size += (a,)
    return c_size


def expanded_size(expand_size, orig_size):
    """Returns the expanded size given two sizes"""
    # strip leading 1s from original size
    if not expand_size:
        return orig_size
    if orig_size == (1,):
        return expand_size
    else:
        return expand_size + orig_size


def batch_sum(v, sample_dims=None, batch_dims=None):
    if sample_dims is None:
        sample_dims = ()
    elif isinstance(sample_dims, int):
        sample_dims = (sample_dims,)
    if batch_dims is None:
        batch_dims = ()
    elif isinstance(batch_dims, int):
        batch_dims = (batch_dims,)
    assert set(sample_dims).isdisjoint(set(batch_dims))
    keep_dims = tuple(sorted(set(sample_dims).union(set(batch_dims))))
    v_sum = partial_sum(v, keep_dims=keep_dims)
    # ToDo: Can we do this more elegantly?
    if len(keep_dims) == 2 and sample_dims[0] > batch_dims[0]:
        return v_sum.permute(1, 0)
    else:
        return v_sum


def partial_sum(v, keep_dims=[]):
    """Sums variable or tensor along all dimensions except those specified
    in `keep_dims`"""
    if len(keep_dims) == 0:
        return v.sum()
    else:
        keep_dims = sorted(keep_dims)
        drop_dims = list(set(range(v.dim())) - set(keep_dims))
        result = v.permute(*(keep_dims + drop_dims))
        size = result.size()[:len(keep_dims)] + (-1,)
        return result.contiguous().view(size).sum(-1)


def log_mean_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().mean(dim, keepdim).log()
    """
    if dim is None:
        s = value.view(-1).size(0)
    else:
        s = value.size(dim)
    return log_sum_exp(value, dim, keepdim) - math.log(s)


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def mkdirs(path):

    if not os.path.exists(path):
        os.makedirs(path)


def grid2gif(img_dir, out_gif, delay=100, duration=0.1):
    '''
    make (moving) GIF from images
    '''

    if True:  # os.name=='nt':

        fnames = [ \
            str(os.path.join(img_dir, f)) for f in os.listdir(img_dir) \
            if ('jpg' in f)]

        fnames.sort()

        images = []
        for filename in fnames:
            images.append(imageio.imread(filename))

        imageio.mimsave(out_gif, images, duration=duration)

    else:  # os.name=='posix'

        img_str = str(os.path.join(img_dir, '*.jpg'))
        cmd = 'convert -delay %s -loop 0 %s %s' % (delay, img_str, out_gif)
        subprocess.call(cmd, shell=True)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def kaiming_init(m, seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def xavier_init(m, seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def transform(image, resize=None):
    from PIL import Image

    if len(image.shape) ==3:
        image = np.transpose(image, (1, 2, 0))
        image = Image.fromarray(image, mode='RGB')
    else:
        image = Image.fromarray(image, mode='L')
    if resize:
        image = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])(image)
    else:
        image = transforms.Compose([
            transforms.ToTensor()
        ])(image)
    return image


def logsumexp(x, dim=None, keepdim=False):
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim, keepdim=True)
    x = torch.where(
        (xm == float('inf')) | (xm == float('-inf')),
        xm,
        xm + torch.log(torch.sum(torch.exp(x - xm), dim, keepdim=True)))
    return x if keepdim else x.squeeze(dim)


def apply_poe(use_cuda, mu_sharedA, std_sharedA, mu_sharedB=None, std_sharedB=None):
    '''
    induce zS = encAB(xA,xB) via POE, that is,
        q(zI,zT,zS|xI,xT) := qI(zI|xI) * qT(zT|xT) * q(zS|xI,xT)
            where q(zS|xI,xT) \propto p(zS) * qI(zS|xI) * qT(zS|xT)
    '''
    EPS = 1e-9
    ZERO = torch.zeros(std_sharedA.shape)
    if use_cuda:
        ZERO = ZERO.cuda()

    logvar_sharedA = torch.log(std_sharedA ** 2 + EPS)

    if mu_sharedB is not None and std_sharedB is not None:
        logvar_sharedB = torch.log(std_sharedB ** 2 + EPS)
        logvarS = -logsumexp(
            torch.stack((ZERO, -logvar_sharedA, -logvar_sharedB), dim=2),
            dim=2
        )
    else:
        logvarS = -logsumexp(
            torch.stack((ZERO, -logvar_sharedA), dim=2),
            dim=2
        )
    stdS = torch.sqrt(torch.exp(logvarS))

    if mu_sharedB is not None and std_sharedB is not None:
        muS = (mu_sharedA / (std_sharedA ** 2 + EPS) +
               mu_sharedB / (std_sharedB ** 2 + EPS)) * (stdS ** 2)
    else:
        muS = (mu_sharedA / (std_sharedA ** 2 + EPS)) * (stdS ** 2)

    return muS, stdS


def apply_poe18(use_cuda, mu_shared, std_shared):
    '''
    induce zS = encAB(xA,xB) via POE, that is,
        q(zI,zT,zS|xI,xT) := qI(zI|xI) * qT(zT|xT) * q(zS|xI,xT)
            where q(zS|xI,xT) \propto p(zS) * qI(zS|xI) * qT(zS|xT)
    '''
    EPS = 1e-9
    ZERO = torch.zeros(std_shared[0].shape)
    if use_cuda:
        ZERO = ZERO.cuda()

    logvar_shared = [ZERO]
    for std in std_shared:
        logvar_shared.append(-torch.log(std ** 2) + EPS)

    logvarS = -torch.logsumexp(
        torch.stack(logvar_shared, dim=2),
        dim=2
    )
    stdS = torch.sqrt(torch.exp(logvarS))
    muS = 0
    for i in range(len(mu_shared)):
        muS += mu_shared[i] / (std_shared[i] ** 2)
    muS = muS * (stdS ** 2)

    return muS, stdS



class DataGather(object):
    '''
    create (array)lists, one for each category, eg,
      self.data['recon'] = [2.3, 1.5, 0.8, ...],
      self.data['kl'] = [0.3, 1.8, 2.2, ...],
      self.data['acc'] = [0.3, 0.4, 0.5, ...], ...
    '''

    def __init__(self, *args):
        self.keys = args
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return {arg: [] for arg in self.keys}

    def insert(self, **kwargs):
        for key in kwargs.keys():
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()

import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def ProductOfExperts(mu, logvar, eps=1e-8):
    var = torch.exp(logvar) + eps
    # precision of i-th Gaussian expert at point x
    T = 1. / (var + eps)
    pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
    pd_var = 1. / torch.sum(T, dim=0)
    pd_logvar = torch.log(pd_var + eps)
    return pd_mu, pd_logvar


from torch.autograd import Variable


def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu = Variable(torch.zeros(size))
    logvar = Variable(torch.log(torch.ones(size)))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar
