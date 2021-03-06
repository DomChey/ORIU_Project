"""
Custom datasets and custom dataloader functions for our selection of the MPII Human Pose Dataset
@author: Dominique Cheray
"""

# necessary imports
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from PIL import Image


class SportsDataTrain(Dataset):
    """Dataset for the trainig images"""

    def __init__(self, transformations):
        """Init images and labels"""
        self.x = []
        self.y = []
        self.transformations = transformations
        # load the images and labels
        imagefile = open("images/train_images.txt")
        images = imagefile.read().splitlines()
        imagefile.close()
        labelsfile = open("images/train_labels.txt")
        labels = labelsfile.read().splitlines()
        labelsfile.close()

        # store images and labels in xs and ya
        for i in range(len(labels)):
            self.x.append("images/" + images[i])
            self.y.append(np.float32(labels[i]))

    def __getitem__(self, index):
        """Override to give pytorch access to images on the dataset"""

        # laod the image and apply transforms
        img = Image.open(self.x[index])
        img = img.convert('RGB')
        img = self.transformations(img)
        # convert label to tensor
        label = torch.from_numpy(np.asarray(self.y[index])).long()
        return img, label

    def __len__(self):
        """override to give pytorch the size of the dataset"""
        return len(self.x)


class SportsDataTest(Dataset):
    """Dataset for the testing images"""

    def __init__(self, transformations):
        """Init images and labels"""
        self.x = []
        self.y = []
        self.transformations = transformations
        # load the images and labels
        imagefile = open("images/test_images.txt")
        images = imagefile.read().splitlines()
        imagefile.close()
        labelsfile = open("images/test_labels.txt")
        labels = labelsfile.read().splitlines()
        labelsfile.close()

        # store images and labels in xs and ya
        for i in range(len(labels)):
            self.x.append("images/" + images[i])
            self.y.append(np.float32(labels[i]))

    def __getitem__(self, index):
        """Override to give pytorch access to images on the dataset"""

        # laod the image and apply transforms
        img = Image.open(self.x[index])
        img = img.convert('RGB')
        img = self.transformations(img)
        # convert label to tensor
        label = torch.from_numpy(np.asarray(self.y[index])).long()
        return img, label

    def __len__(self):
        """override to give pytorch the size of the dataset"""
        return len(self.x)


def get_train_and_validation_loader(batch_size, augment, use_cuda, 
                                    random_seed=0, valid_size=0.1, shuffle=True,):
    """Useful function that loads and returns train and validation dataloader
    for the datset

    Args:
        batch_size: desired batch size
        augment: whether to apply data augmentation to the train split, will
                 not be applied to the validation split
        random_seed; fix it for reproducability default is 0 which means he will not be fixed
        valid_size: percentage of data used for the validation set. Should be
                    a float between 0 and 1
        shuffle: whether to shuffle the train/validation indices
        use_cuda: whether cuda is available or not. Set it to true if cuda is used
    Returns:
       train_loader: dataloader for the training set
       validation_loader: dataloader for the validation set
    """
    resize = transforms.Resize((224, 224))
    # define transforms for validation and train set
    validation_transforms = transforms.Compose([
        resize,
        transforms.ToTensor()
    ])
    # if desired apply data augmentation to the training data
    if augment:
        train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.FiveCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])
    else:
        train_transforms = transforms.Compose([
            resize,
            transforms.ToTensor()
        ])

    # now load the dataset
    train_set = SportsDataTrain(train_transforms)
    validation_set = SportsDataTrain(validation_transforms)
    # get length of dataset and use it to create the indices for train and validation set
    num_train = len(train_set)
    indices = list(range(num_train))

    # determine where to split for validation
    split_index = (np.floor(valid_size * num_train)).astype(int)
    # shuffle indices if asked for
    if shuffle:
        if random_seed != 0:
            np.random.seed(random_seed)
        np.random.shuffle(indices)
    # now split the indices for training and validation
    train_indices = indices[split_index:]
    valid_indices = indices[:split_index]

    # create random subsampler for validation and test set that sample from the given indices
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(valid_indices)

    # now create the two dataloader and give them the subsampler
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              sampler=train_sampler, num_workers=1,
                              pin_memory=use_cuda)
    validation_loader = DataLoader(validation_set, batch_size=batch_size,
                                   sampler=validation_sampler, num_workers=1,
                                   pin_memory=use_cuda)
    # WINDOWS ONLY
#    train_loader = DataLoader(train_set, batch_size=batch_size,
#                              sampler=train_sampler, num_workers=0,
#                              pin_memory=use_cuda)
#    validation_loader = DataLoader(validation_set, batch_size=batch_size,
#                                   sampler=validation_sampler, num_workers=0,
#                                   pin_memory=use_cuda)
#
    return train_loader, validation_loader


def get_test_loader(batch_size, use_cuda, shuffle=True):
    """Useful function that loads and returns a test dataloader for the test dataset

    Args:
        batch_size: desired batch size
        use_cude: whether to use cuda or not
        shuffle: whether to shuffle the dataset or not

    Returns:
        test_loader: dataloader for the test set
    """
    resize = transforms.Resize((224, 224))
    # define transforms
    transformations = transforms.Compose([
        resize,
        transforms.ToTensor()
    ])
    # load the dataset
    test_set = SportsDataTest(transformations)
    # create the dataloader
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle,
                             num_workers=1, pin_memory=use_cuda)

    return test_loader
