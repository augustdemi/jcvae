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
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
# from utils import transform
from PIL import Image


class datasets(Dataset):
    def __init__(self, path, primary_attr_idx, train=True):
        path = '../../data/cub/CUB_200_2011/CUB_200_2011/'
        self.filepaths, self.attributes, self.labels = load_data(train, path, primary_attr_idx)
        self.path = path


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

        img = pil_loader(self.path + '/images/' + img_path)
        img = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])(img)

        # imgshow(img)
        # for i in range(self.labels.shape[0]):
        #     if self.labels[i] == 163:
        #         print(i)

        label = torch.tensor(label - 1, dtype=torch.int64)
        # attr = torch.FloatTensor(attr)
        return img, attr, label

    def __len__(self):
        return len(self.labels)


def load_data(train, path, primary_attr_idx):
    train_classes = np.genfromtxt(path + "attributes/trainvalids.txt", delimiter='\n', dtype=int)
    imgid_label = np.array([int(elt.split(' ')[1]) for elt in
                            np.genfromtxt(path + "attributes/image_class_labels.txt", delimiter='\n', dtype=str)])

    attributes = []
    if train:
        # trainset - all modalities: img path, attr, label
        tr_vec_attr = pickle.load(open(path + "attributes/vec_attr_trainval.pkl", "rb"))
        for key in tr_vec_attr.keys():
            attributes.append([tr_vec_attr[key][i] / tr_vec_attr[key][i].sum() for i in primary_attr_idx])
        tr_imgidx = np.array([i for i in range(len(imgid_label)) if imgid_label[i] in train_classes])
        labels = list(imgid_label[tr_imgidx])
        filepaths = np.array(
            [elt.split(' ')[1] for elt in np.genfromtxt(path + "attributes/images.txt", delimiter='\n', dtype=str)])[
            tr_imgidx]
        filepaths = list(filepaths)

        # testset - only attributes and labels
        # te_vec_attr = pickle.load(open(path + "attributes/vec_attr_test.pkl", "rb"))
        # for key in te_vec_attr.keys():
        #     attributes.append([te_vec_attr[key][i] for i in primary_attr_idx])
        # te_imgidx = np.array([i for i in range(len(imgid_label)) if imgid_label[i] not in train_classes])
        # labels.extend(list(imgid_label[te_imgidx]))
        # filepaths.extend(list(np.array(
        #     [elt.split(' ')[1] for elt in np.genfromtxt(path + "attributes/images.txt", delimiter='\n', dtype=str)])[
        #     te_imgidx]))
        # filepaths.extend([None] * te_imgidx.shape[0])
    else:
        te_vec_attr = pickle.load(open(path + "attributes/vec_attr_test.pkl", "rb"))
        for key in te_vec_attr.keys():
            attributes.append([te_vec_attr[key][i] / te_vec_attr[key][i].sum() for i in primary_attr_idx])
        te_imgidx = np.array([i for i in range(len(imgid_label)) if imgid_label[i] not in train_classes])
        labels = imgid_label[te_imgidx]
        filepaths = np.array(
            [elt.split(' ')[1] for elt in np.genfromtxt(path + "attributes/images.txt", delimiter='\n', dtype=str)])[
            te_imgidx]
    return filepaths, attributes, labels


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
