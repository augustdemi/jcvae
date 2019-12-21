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
import os
from scipy import io, misc
from torchvision import transforms
from torch.utils.data.dataset import Dataset
# from utils import transform
from PIL import Image

import random

class datasets(Dataset):
    def __init__(self, path, primary_attr_idx, train=True, crop=None):
        self.path = path
        self.crop = crop

        self.filepaths, self.attributes, self.labels, self.boxes = load_data(train, path, primary_attr_idx)

        n_data = len(self.filepaths)
        n_tr_data = int(n_data * 0.7)

        total_idx = list(range(n_data))
        random.seed(324)
        random.shuffle(total_idx)

        tr_idx = total_idx[:n_tr_data]
        te_idx = total_idx[n_tr_data:]

        if train:
            self.filepaths = np.array(self.filepaths)[tr_idx]
            self.attributes = np.array(self.attributes)[tr_idx]
            self.labels = np.array(self.labels)[tr_idx]
            self.boxes = np.array(self.boxes)[tr_idx]
        else:
            self.filepaths = np.array(self.filepaths)[te_idx]
            self.attributes = np.array(self.attributes)[te_idx]
            self.labels = np.array(self.labels)[te_idx]
            self.boxes = np.array(self.boxes)[te_idx]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img_path, attr, label, box = self.filepaths[index], self.attributes[index], self.labels[index], self.boxes[
            index]

        img = pil_loader(self.path + '/images/' + img_path)
        crop_img = self._read(img, index)

        crop_img = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])(Image.fromarray(crop_img))
        # imgshow(crop_img)

        label = torch.tensor(label - 1, dtype=torch.int64)
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
        # if self._target_size is not None:
        #     image = misc.imresize(image, self._target_size)
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


def load_data(train, path, primary_attr_idx):
    boxes = []
    for line in open(path + 'attributes/bounding_boxes.txt', 'r'):
        box = []
        for val in line.split(' ')[1:]:
            box.append(float(val))
        boxes.append(box)
    boxes = np.array(boxes)
    # boxes = [box for box, val in zip(boxes, is_selected) if val]
    train_classes = np.genfromtxt(path + "attributes/trainvalids.txt", delimiter='\n', dtype=int)
    imgid_label = np.array([int(elt.split(' ')[1]) for elt in
                            np.genfromtxt(path + "attributes/image_class_labels.txt", delimiter='\n', dtype=str)])
    filepaths = np.array(
        [elt.split(' ')[1] for elt in np.genfromtxt(path + "attributes/images.txt", delimiter='\n', dtype=str)])


    attributes = []
    tr_vec_attr = pickle.load(open(path + "attributes/vec_attr_trainval.pkl", "rb"))
    for key in tr_vec_attr.keys():
        bin_attr = []
        for i in primary_attr_idx:
            bin_attr.extend(tr_vec_attr[key][i][:-1])
        attributes.append(bin_attr)

    te_vec_attr = pickle.load(open(path + "attributes/vec_attr_test.pkl", "rb"))
    for key in te_vec_attr.keys():
        bin_attr = []
        for i in primary_attr_idx:
            bin_attr.extend(te_vec_attr[key][i][:-1])
        attributes.append(bin_attr)

    tr_imgidx = np.array([i for i in range(len(imgid_label)) if imgid_label[i] in train_classes])
    te_imgidx = np.array([i for i in range(len(imgid_label)) if imgid_label[i] not in train_classes])

    filepaths = list(filepaths[tr_imgidx]) + list(filepaths[te_imgidx])
    labels = list(imgid_label[tr_imgidx]) + list(imgid_label[te_imgidx])
    boxes = list(boxes[tr_imgidx]) + list(boxes[te_imgidx])

    return filepaths, attributes, labels, boxes


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
