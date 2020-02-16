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

import pickle
import numpy as np

import torch
import h5py
from scipy import io, misc
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import Normalizer


import random


class datasets(Dataset):
    def __init__(self, path, primary_attr_idx, train=True, crop=None, val=False):
        self.filepaths, self.attributes, self.labels, self.boxes = load_data(train, path, primary_attr_idx)
        self.path = path
        self.crop = crop

        if train:
            n_data = self.filepaths.shape[0]
            n_tr_data = int(n_data * 0.8)

            total_idx = list(range(n_data))
            random.seed(324)
            random.shuffle(total_idx)

            tr_idx = total_idx[:n_tr_data]
            val_idx = total_idx[n_tr_data:]
            if val:
                self.filepaths = self.filepaths[val_idx]
                self.labels = self.labels[val_idx]
                self.boxes = self.boxes[val_idx]
            else:
                self.filepaths = self.filepaths[tr_idx]
                self.labels = self.labels[tr_idx]
                self.boxes = self.boxes[tr_idx]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # index = 2233
        # img_idx: 8055
        # paht: 138.Tree_Swallow / Tree_Swallow_0060_134961.jpg
        # label:138
        # bding box: 13.0 39.0 476.0 458.0
        # attr: checked
        img_feat, label, box = self.filepaths[index], self.labels[index], self.boxes[
            index]

        label = torch.tensor(label - 1, dtype=torch.int64)
        # attr = torch.FloatTensor(attr)
        return img_feat, self.attributes[label], label

    def __len__(self):
        return len(self.labels)


def load_data(train, path, primary_attr_idx):
    boxes = []
    for line in open(path + 'bounding_boxes.txt', 'r'):
        box = []
        for val in line.split(' ')[1:]:
            box.append(float(val))
        boxes.append(box)
    boxes = np.array(boxes)
    imgid_label = np.array([int(elt.split(' ')[1]) for elt in
                            np.genfromtxt(path + "image_class_labels.txt", delimiter='\n', dtype=str)])

    if train:
        hf = h5py.File(path + 'feature/trainval_feat.h5', 'r')
    else:
        hf = h5py.File(path + 'feature/test_feat.h5', 'r')

    attributes = np.array([elt.split(' ') for elt in
                           np.genfromtxt(path + "attributes/class_attribute_labels_continuous.txt", delimiter='\n',
                                         dtype=str)])

    attributes = attributes.astype(float)
    normalizer = Normalizer(norm='l2').fit(attributes)
    attributes = normalizer.transform(attributes)

    img_feat = np.array(hf['feature'])
    index = np.array(hf['index'])
    labels = list(imgid_label[index])
    boxes = list(boxes[index])

    return img_feat, attributes, np.array(labels), np.array(boxes)
