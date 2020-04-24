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

    def __init__(self, root, train=True):
        self.root = os.path.expanduser(root)
        self.img_upper, self.img_bottom, self.label = load_mnist(train)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_upper, img_bottom = self.img_upper[index], self.img_bottom[index]

        img_upper = transforms.ToTensor()(img_upper)
        img_bottom = transforms.ToTensor()(img_bottom)

        # imgshow(img_upper)
        # imgshow(img_bottom)

        return img_upper, img_bottom

    def __len__(self):
        return self.img_upper.shape[0]


def load_mnist(train):
    train_loader = torch.utils.data.DataLoader(
        dset.MNIST(root='../../data/mnist', train=True, download=True, transform=transforms.ToTensor()))

    test_loader = torch.utils.data.DataLoader(
        dset.MNIST(root='../../data/mnist', train=False, download=True, transform=transforms.ToTensor()))

    if train:
        img = train_loader.dataset.train_data.numpy()[:50000]
        label = train_loader.dataset.train_labels.numpy()[:50000]
    else:
        img = test_loader.dataset.test_data.numpy()
        label = test_loader.dataset.test_labels.numpy()

    return img[:, :14, :], img[:, 14:, :], label


# def pil_loader(path):
#     # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGB')


def imgshow(img):
    from torchvision.transforms import ToPILImage
    import matplotlib.pyplot as plt

    to_img = ToPILImage()
    plt.imshow(to_img(img))
    print(img.size())
    print("max: {}, min: {}".format(np.max(img.numpy()), np.min(img.numpy())))
