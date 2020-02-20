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
from PIL import Image

import random


class datasets(Dataset):
    def __init__(self, path, train=True, crop=None, val=False):
        self.filepaths, self.attributes, self.labels, self.boxes = load_data(train, path)
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

        img_path, label, box = self.filepaths[index], self.labels[index], self.boxes[
            index]
        img = pil_loader(self.path + '/images/' + img_path)
        crop_img = self._read(img, index)

        crop_img = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])(Image.fromarray(crop_img))

        label = torch.tensor(label - 1, dtype=torch.int64)
        attr = torch.tensor(self.attributes[label], dtype=torch.float)
        return crop_img, attr, label

    def __len__(self):
        return len(self.labels)

    def _read(self, img, index):
        raw_dim = img.size
        img = np.asarray(img)
        if self.crop is not None:
            xmin, ymin, xmax, ymax = self._get_cropped_coordinates(index, raw_dim)
        else:
            xmin, ymin, xmax, ymax = 0, 0, raw_dim[0], raw_dim[1]
        crop_img = img[ymin:ymax, xmin:xmax]
        return crop_img

    def _get_cropped_coordinates(self, index, raw_dim):
        box = self.boxes[index]
        x, y, width, height = box
        centerx = x + width / 2.
        centery = y + height / 2.
        xoffset = width * self.crop / 2.
        yoffset = height * self.crop / 2.
        xmin = max(int(centerx - xoffset + 0.5), 0)
        ymin = max(int(centery - yoffset + 0.5), 0)
        xmax = min(int(centerx + xoffset + 0.5), raw_dim[0] - 1)
        ymax = min(int(centery + yoffset + 0.5), raw_dim[1] - 1)
        # if xmax - xmin <= 0 or ymax - ymin <= 0:
        #     raise ValueError, "The cropped bounding box has size 0."
        return xmin, ymin, xmax, ymax


def load_data(train, path):
    boxes = []
    for line in open(path + 'bounding_boxes.txt', 'r'):
        box = []
        for val in line.split(' ')[1:]:
            box.append(float(val))
        boxes.append(box)
    boxes = np.array(boxes)
    imgid_label = np.array([int(elt.split(' ')[1]) for elt in
                            np.genfromtxt(path + "image_class_labels.txt", delimiter='\n', dtype=str)])

    trainval_classes = np.genfromtxt(path + "cvpr2016/trainvalids.txt", delimiter='\n', dtype=int)
    index = []
    if train:
        index = np.array([i for i in range(len(imgid_label)) if imgid_label[i] in trainval_classes])
    else:
        index = np.array([i for i in range(len(imgid_label)) if imgid_label[i] not in trainval_classes])

    filepaths = np.array(
        [elt.split(' ')[1] for elt in np.genfromtxt(path + "images.txt", delimiter='\n', dtype=str)])[
        index]

    attributes = np.array([elt.split(' ') for elt in
                           np.genfromtxt(path + "attributes/class_attribute_labels_continuous.txt", delimiter='\n',
                                         dtype=str)])
    attributes = attributes.astype(float)
    normalizer = Normalizer(norm='l2').fit(attributes)
    attributes = normalizer.transform(attributes)

    labels = list(imgid_label[index])
    boxes = list(boxes[index])

    return filepaths, attributes, np.array(labels), np.array(boxes)


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
