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
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
# from utils import transform
from PIL import Image


class datasets(Dataset):
    def __init__(self, train=True):
        self.filepaths, self.attributes, self.labels = load_data(train)


        # dataset = torchvision.datasets.ImageFolder(
        #     root= self.filepaths,
        #     transform=transforms.Compose([
        #     transforms.Resize(64),
        #     transforms.ToTensor()
        # ])
        # )
        # path = '../../data/awa/Animals_with_Attributes2/'
        # attributes = get_attr(dataset.targets, path)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path, attr, label = self.filepaths[index], self.attributes[index], self.labels[index]

        img = pil_loader(img_path)
        img = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])(img)
        # imgshow(img)
        label = torch.tensor(label - 1, dtype=torch.int64)
        attr = torch.FloatTensor(attr)
        return img, attr, label

    def __len__(self):
        # return self.input_a.size(0)
        return len(self.labels)


def load_data(train):
    path = '../../data/awa/Animals_with_Attributes2/'
    img_path = 'JPEGImages/'

    # meta
    train_classes = np.genfromtxt(path + 'trainclasses.txt', delimiter='\n', dtype=str)
    # class_meta = np.genfromtxt(path + 'classes.txt', delimiter='\n', dtype=str)
    # test_classes = np.genfromtxt(path + 'testclasses.txt', delimiter='\n', dtype=str)
    attr_meta = list(np.genfromtxt(path + 'predicate-matrix-binary.txt', delimiter='\n', dtype=str))
    for i in range(len(attr_meta)):
        attr_meta[i] = attr_meta[i].split(' ')
    attr_meta = np.array(attr_meta).astype(float)
    # attr_meta = torch.tensor(attr_meta, dtype=float)


    filenames = np.genfromtxt(path + 'AwA2-filenames.txt', delimiter='\n', dtype=str)
    labels = np.genfromtxt(path + 'AwA2-labels.txt', delimiter='\n', dtype=int)
    # features = list(np.genfromtxt(path + 'AwA2-features.txt', delimiter='\n', dtype=str))
    # for i in range(len(features)):
    #     features[i] = features[i].split(' ')
    # features = np.array(features).astype(float)


    tr_idx = []
    te_idx = []
    attributes = []
    filepaths = []
    for i in range(len(labels)):
        attributes.append(attr_meta[labels[i] - 1])
        filename = filenames[i]
        class_name = filename.split('_')[0]
        filepaths.append(os.path.join(path, img_path, class_name, filename))
        if class_name in train_classes:
            tr_idx.append(i)
        else:
            te_idx.append(i)
    attributes = np.array(attributes)
    filepaths = np.array(filepaths)

    if train:
        return filepaths[tr_idx], attributes[tr_idx], labels[tr_idx]
    else:
        return filepaths[te_idx], attributes[te_idx], labels[te_idx]


def get_attr(labels, path):
    attr_meta = list(np.genfromtxt(path + 'predicate-matrix-binary.txt', delimiter='\n', dtype=str))
    for i in range(len(attr_meta)):
        attr_meta[i] = attr_meta[i].split(' ')
    attr_meta = np.array(attr_meta).astype(float)

    attributes = []
    for i in range(len(labels)):
        attributes.append(attr_meta[labels[i]])
    return np.array(attributes)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def imgshow(img):
    from torchvision.transforms import ToPILImage
    import matplotlib.pyplot as plt

    to_img = ToPILImage()
    plt.imshow(to_img(img))
    print(img.size())
    print("max: {}, min: {}".format(np.max(img.numpy()), np.min(img.numpy())))
