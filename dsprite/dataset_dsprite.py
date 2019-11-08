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

import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import os

class Position(Dataset):

    def __init__(self):
        # self.root = os.path.expanduser(root)
        self.input_a, self.input_b = load_dsprite()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        a_img, b_img =  self.input_a[index], self.input_b[index]
        return a_img, b_img

    def __len__(self):
        return self.input_a.size(0)
        # return len(self.pair_label)


def load_dsprite():

    # latent_values = np.load(os.path.join('../../data/'
    #                                      'dsprites-dataset', 'latents_values.npy'), encoding='latin1')
    # latent_values = latent_values[:, [1, 2, 3, 4, 5]]
    # # latent values (actual values);(737280 x 5)
    # latent_classes = np.load(os.path.join('../../data/',
    #                                       'dsprites-dataset', 'latents_classes.npy'), encoding='latin1')
    # latent_classes = latent_classes[:, [1, 2, 3, 4, 5]]

    root = os.path.join('../../data/', 'dsprites-dataset/imgs.npy')
    imgs = np.load(root, encoding='latin1')
    # data = torch.from_numpy(imgs).unsqueeze(1).float()

    one_shape = int(imgs.shape[0] / 3)

    imgsA = list(imgs[:one_shape])
    imgsA.extend(list(imgs[2*one_shape:]))
    imgsB = list(imgs[one_shape:2*one_shape])
    imgsB.extend(imgsB)

    imgsA = torch.from_numpy(np.array(imgsA)).unsqueeze(1).float()
    imgsB = torch.from_numpy(np.array(imgsB)).unsqueeze(1).float()

    return imgsA, imgsB


