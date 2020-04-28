"""
This script generates a dataset similar to the MultiMNIST dataset
described in [1]. However, we remove any translation.

[1] Eslami, SM Ali, et al. "Attend, infer, repeat: Fast scene
understanding with generative models." Advances in Neural Information
Processing Systems. 2016.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np

import torch
import torchvision.datasets as dset
from torchvision import transforms
from torch.utils.data.dataset import Dataset


# from utils import transform


class DIGIT(Dataset):
    """Images with 0 to 4 digits of non-overlapping MNIST numbers.

    @param root: string
                 path to dataset root
    @param train: boolean [default: True]
           whether to return training examples or testing examples
    @param transform: ?torchvision.Transforms
                      optional function to apply to training inputs
    @param target_transform: ?torchvision.Transforms
                             optional function to apply to training outputs
    """
    processed_folder = 'digit'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, aug=False):
        self.root = os.path.expanduser(root)
        self.input_a, self.input_b, self.a_idx, self.b_idx, self.label = make_dataset_fixed(train)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        mnist_img, svhn_img, label = self.input_a[self.a_idx[index]], self.input_b[self.b_idx[index]], self.label[index]

        mnist_img = transforms.ToTensor()(mnist_img)
        svhn_img = transforms.ToTensor()(svhn_img)
        svhn_img = torch.transpose(svhn_img, 0, 1)

        return mnist_img, svhn_img

    def __len__(self):
        # return self.input_a.size(0)
        return len(self.label)


def load_mnist():
    train_loader = torch.utils.data.DataLoader(
        dset.MNIST(root='../../data/mnist', train=True, download=True, transform=transforms.ToTensor()))

    test_loader = torch.utils.data.DataLoader(
        dset.MNIST(root='../../data/mnist', train=False, download=True, transform=transforms.ToTensor()))

    train_data = {
        'imgs': train_loader.dataset.train_data.numpy(),
        'labels': train_loader.dataset.train_labels.numpy()
    }

    test_data = {
        'imgs': test_loader.dataset.test_data.numpy(),
        'labels': test_loader.dataset.test_labels.numpy()
    }
    train_class_idx = {}
    test_class_idx = {}
    for i in range(10):
        train_class_idx.update({i: []})
        test_class_idx.update({i: []})
    for i in range(len(train_data['labels'])):
        train_class_idx[train_data['labels'][i]].append(i)
    for i in range(len(test_data['labels'])):
        test_class_idx[test_data['labels'][i]].append(i)
    train_data['class_idx'] = train_class_idx
    test_data['class_idx'] = test_class_idx
    return train_data, test_data


def load_svhn():
    train_loader = torch.utils.data.DataLoader(
        dset.SVHN(root='../../data/svhn', split='train', download=True, transform=transforms.ToTensor()))

    test_loader = torch.utils.data.DataLoader(
        dset.SVHN(root='../../data/svhn', split='test', download=True, transform=transforms.ToTensor()))

    train_data = {
        'imgs': train_loader.dataset.data,
        'labels': train_loader.dataset.labels
    }

    test_data = {
        'imgs': test_loader.dataset.data,
        'labels': test_loader.dataset.labels
    }

    train_class_idx = {}
    test_class_idx = {}
    for i in range(10):
        train_class_idx.update({i: []})
        test_class_idx.update({i: []})
    for i in range(len(train_data['labels'])):
        train_class_idx[train_data['labels'][i]].append(i)
    for i in range(len(test_data['labels'])):
        test_class_idx[test_data['labels'][i]].append(i)
    train_data['class_idx'] = train_class_idx
    test_data['class_idx'] = test_class_idx
    return train_data, test_data


def match_label(modalA, modalB):
    a_idx, b_idx, labels = [], [], []
    for i in range(len(modalA['class_idx'])):
        class_idxA = modalA['class_idx'][i]
        class_idxB = modalB['class_idx'][i]
        print('label {} len A, B: {}, {}'.format(i, len(class_idxA), len(class_idxB)))

        diff = len(class_idxA) - len(class_idxB)
        if diff > 0:
            a_idx.extend(class_idxA)
            b_idx.extend(class_idxB)
            while diff > 0:
                if diff > len(class_idxB):
                    np.random.seed(0)
                    np.random.shuffle(class_idxB)
                    b_idx.extend(class_idxB)
                else:
                    np.random.seed(0)
                    np.random.shuffle(class_idxB)
                    b_idx.extend(class_idxB[:diff])
                diff -= len(class_idxB)
        else:
            diff = -diff
            a_idx.extend(class_idxA)
            b_idx.extend(class_idxB)

            while diff > 0:
                if diff > len(class_idxA):
                    np.random.seed(0)
                    np.random.shuffle(class_idxA)
                    a_idx.extend(class_idxA)
                else:
                    np.random.seed(0)
                    np.random.shuffle(class_idxA)
                    a_idx.extend(class_idxA[:diff])
                diff -= len(class_idxA)

        labels.extend([i] * max(len(class_idxA), len(class_idxB)))
    return a_idx, b_idx, labels


def make_dataset_fixed(train):
    np.random.seed(681307)
    trainA, testA = load_mnist()
    trainB, testB = load_svhn()

    if train:
        modalA = trainA
        modalB = trainB
    else:
        modalA = testA
        modalB = testB
    a_idx, b_idx, labels = match_label(modalA, modalB)

    return modalA['imgs'], modalB['imgs'], a_idx, b_idx, labels
