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
from torchvision import datasets, transforms
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

    def __init__(self, root, train=True):
        self.root = os.path.expanduser(root)
        self.img, self.label = load_mnist(train)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.img[index], self.label[index]

        img = transforms.ToTensor()(img)

        return img, label

    def __len__(self):
        return len(self.label)


def load_mnist(train):
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN('../../data/svhn', split='train', download=True,
                      transform=transforms.ToTensor()))
    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN('../../data/svhn', split='test', download=True,
                      transform=transforms.ToTensor()))

    if train:
        img = train_loader.dataset.train_data.numpy()[:60000]
        label = train_loader.dataset.train_labels.numpy()[:60000]
    else:
        img = test_loader.dataset.test_data.numpy()
        label = test_loader.dataset.test_labels.numpy()

    return img, label
