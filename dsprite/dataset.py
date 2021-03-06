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

import sys
sys.path.append('../')
import probtorch

class Position(Dataset):
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

    def __init__(self):
        # self.root = os.path.expanduser(root)
        self.input_a, self.input_b, self.a_idx, self.b_idx, self.pair_label = make_dataset_fixed()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        a_img, b_img, pair_label = self.input_a['imgs'][self.a_idx[index]], self.input_b['imgs'][self.b_idx[index]], self.pair_label[index]
        a_img = torch.tensor(a_img, dtype=torch.float32).unsqueeze(0)
        return a_img, b_img, pair_label

    def __len__(self):
        # return self.input_a.size(0)
        return len(self.pair_label)

    def get_3dface(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        b_img = self.input_b['imgs'][index]

        return b_img

    def get_pair(self, index):
        label = self.label[index]
        sum = np.array(self.per_class_n_pairs).cumsum()

        if label != 0:
            within_class_index = index - sum[label-1]
        else:
            within_class_index = index
        a_idx = int(within_class_index / len(self.class_idxB[label]))
        b_idx =  within_class_index % len(self.class_idxB[label])
        a_idx = self.class_idxA[label][a_idx]
        b_idx = self.class_idxB[label][b_idx]
        return a_idx, b_idx


def load_3dface():
    # latent factor = (id, azimuth, elevation, lighting)
    #   id = {0,1,...,49} (50)
    #   azimuth = {-1.0,-0.9,...,0.9,1.0} (21)
    #   elevation = {-1.0,0.8,...,0.8,1.0} (11)
    #   lighting = {-1.0,0.8,...,0.8,1.0} (11)
    # (number of variations = 50*21*11*11 = 127050)
    latent_classes, latent_values = np.load('../../data/3dfaces/gt_factor_labels.npy')
    root = '../../data/3dfaces/basel_face_renders.pth'
    data = torch.load(root).float().div(255)  # (50x21x11x11x64x64)
    data = data.view(-1, 64, 64).unsqueeze(1)
    n = latent_values.shape[0]
    class_idx = {}
    for i in range(231):
        class_idx.update({i:[]})

    face_pos_pair = {}
    idx = -1
    for i in np.round(np.linspace(1, -1, 21),2):
        for j in np.round(np.linspace(-1, 1, 11),2):
            idx += 1
            face_pos_pair.update({(i,j):idx})

    for i in range(n):
        az, el = np.round(latent_values[i, 1],2), np.round(latent_values[i, 2],2)
        if (az, el) in face_pos_pair.keys():
            idx = face_pos_pair[(az, el)]
            class_idx[idx].append(i)
    return data, class_idx


def load_dsprite():
    dataset_zip = np.load('../../data/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding="latin1")

    imgs = dataset_zip['imgs']
    metadata = dataset_zip['metadata'][()]
    latents_sizes = metadata['latents_sizes'] # [ 1,  3,  6, 40, 32, 32]
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                    np.array([1, ])))

    def latent_to_index(latents):
        return np.dot(latents, latents_bases).astype(int)

    all_other_latents = []
    for i in range(3):
        for j in range(6):
            for k in range(40):
                all_other_latents.append([0, i, j, k])

    pos21 = [int(np.round(elt)) for elt in np.linspace(0,31,21)]
    pos11 = [int(np.round(elt)) for elt in np.linspace(0,30,11)]
    pos11_extend = pos11.copy()
    pos11_extend.extend([elt+1 for elt in pos11])

    class_idx = {}
    for i in range(231):
        class_idx.update({i:[]})
    pos_pair = {}

    idx = -1
    for i in pos21:
        for j in pos11:
            idx += 1
            pos_pair.update({(i,j):idx}) # idx = cardinality of  all possible pairs: 0~230
            pos_pair.update({(i,j+1):idx}) # for y position, j and j+1 are classified into same index in order to use many training data.


    for i in pos21:
        for j in pos11_extend:
            for one_latent in all_other_latents:
                latent = one_latent.copy()
                latent.extend([i, j]) # add posX, posY to (color, type, scale, rotation)
                idx = pos_pair[(i,j)]
                class_idx[idx].append(latent_to_index(latent))
    return imgs, class_idx



def match_label(modalA, modalB):

    a_idx, b_idx, labels = [], [], []
    for i in range(len(modalA['class_idx'])):
        class_idxA = modalA['class_idx'][i]
        class_idxB = modalB['class_idx'][i]
        print('label {} len A, B: {}, {}'.format(i, len(class_idxA), len(class_idxB))) # A=sprite=1440, B=face=550
        a_idx.extend(class_idxA)
        b_idx.extend(class_idxB)
        np.random.shuffle(class_idxB)
        b_idx.extend(class_idxB)
        rest = len(class_idxA) % len(class_idxB)
        np.random.shuffle(class_idxB)
        b_idx.extend(class_idxB[:rest])
        labels.extend([i] * len(class_idxA))
    return np.array(a_idx), np.array(b_idx), np.array(labels)


def make_dataset_fixed():
    np.random.seed(681307)
    imgsA, class_idxA = load_dsprite()
    imgsB, class_idxB = load_3dface()
    modalA = {'imgs': imgsA, 'class_idx': class_idxA}
    modalB = {'imgs': imgsB, 'class_idx': class_idxB}
    a_idx, b_idx, labels = match_label(modalA, modalB)
    return modalA, modalB, a_idx, b_idx, labels

