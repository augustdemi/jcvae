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


class datasets(Dataset):
    def __init__(self, path, train=True, crop=None):
        self.filepaths, self.labels, self.boxes = load_data(train, path)
        self.path = path
        self.crop = crop

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
        img_path, label, box = self.filepaths[index], self.labels[index], self.boxes[
            index]
        # for i in range(len(self.labels)): #16,28 (10,0)
        #     if self.labels[i] == 10:
        #         print(i)
        # img_path = self.filepaths[4288]
        img = pil_loader(self.path + '/images/' + img_path)
        crop_img = self._read(img, index)

        crop_img = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])(Image.fromarray(crop_img))
        # imgshow(crop_img)

        label = torch.tensor(label - 1, dtype=torch.int64)
        # attr = torch.FloatTensor(attr)
        return crop_img, label

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
    # boxes = [box for box, val in zip(boxes, is_selected) if val]
    trainval_classes = np.genfromtxt(path + "cvpr2016/trainids.txt", delimiter='\n', dtype=int)
    imgid_label = np.array([int(elt.split(' ')[1]) for elt in
                            np.genfromtxt(path + "image_class_labels.txt", delimiter='\n', dtype=str)])
    # train_classes = [elt for elt in list(range(1,201)) if elt not in test_classes]

    if train:
        tr_imgidx = np.array([i for i in range(len(imgid_label)) if imgid_label[i] in trainval_classes])
        labels = list(imgid_label[tr_imgidx])
        filepaths = np.array(
            [elt.split(' ')[1] for elt in np.genfromtxt(path + "images.txt", delimiter='\n', dtype=str)])[
            tr_imgidx]
        filepaths = list(filepaths)
        boxes = list(boxes[tr_imgidx])

    else:
        te_imgidx = np.array([i for i in range(len(imgid_label)) if imgid_label[i] not in trainval_classes])
        labels = imgid_label[te_imgidx]
        filepaths = np.array(
            [elt.split(' ')[1] for elt in np.genfromtxt(path + "images.txt", delimiter='\n', dtype=str)])[
            te_imgidx]
        boxes = list(boxes[te_imgidx])
    return filepaths, labels, boxes


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
