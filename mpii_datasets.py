"""
Custom datasets for our selection of the MPII Human Pose Dataset
@author: Dominique Cheray
"""

# necessary imports
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
from PIL import Image


class SportsDataTrain(Dataset):
    """Dataset for the trainig images"""
    __xs = []
    __ys = []

    def __init__(self):
        """Init images and labels"""

        self.transformations = transforms.Compose([transforms.CenterCrop(720),
                                                   transforms.ToTensor()])
        # load the images and labels
        imagefile = open("images/train_images.txt")
        images = imagefile.read().splitlines()
        imagefile.close()
        labelsfile = open("images/train_labels.txt")
        labels = labelsfile.read().splitlines()

        # store images and labels in xs and ya
        for i in range(len(labels)):
            self.__xs.append("images/" + images[i])
            self.__ys.append(np.float32(labels[i]))

    def __getitem__(self, index):
        """Override to give pytorch access to images on the dataset"""

        # laod the image and apply transforms
        img = Image.open(self.__xs[index])
        img = img.convert('RGB')
        img = self.transformations(img)
        # convert label to tensor
        label = torch.from_numpy(np.asarray(self.__ys[index]).reshape([1, 1]))
        return img, label

    def __len__(self):
        """override to give pytorch the size of the dataset"""
        return len(self.__ys)


class SportsDataTest(Dataset):
    """Dataset for the testing images"""
    __xs = []
    __ys = []

    def __init__(self):
        """Init images and labels"""

        self.transformations = transforms.Compose([transforms.CenterCrop(720),
                                                   transforms.ToTensor()])
        # load the images and labels
        imagefile = open("images/test_images.txt")
        images = imagefile.read().splitlines()
        imagefile.close()
        labelsfile = open("images/test_labels.txt")
        labels = labelsfile.read().splitlines()

        # store images and labels in xs and ya
        for i in range(len(labels)):
            self.__xs.append("images/" + images[i])
            self.__ys.append(np.float32(labels[i]))

    def __getitem__(self, index):
        """Override to give pytorch access to images on the dataset"""

        # laod the image and apply transforms
        img = Image.open(self.__xs[index])
        img = img.convert('RGB')
        img = self.transformations(img)
        # convert label to tensor
        label = torch.from_numpy(np.asarray(self.__ys[index]).reshape([1, 1]))
        return img, label

    def __len__(self):
        """override to give pytorch the size of the dataset"""
        return len(self.__ys)
