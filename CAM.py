"""
Implementation of Class Activatoin Mapping
@author: Dominique Cheray
"""

# necessary imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from mpii_datasets import get_train_and_validation_loader, get_test_loader
from googLeNet import GoogLeNet
from PIL import Image


# usefull globals
USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'


def class_activation_mapping(model, imagepath):
    """
    Class Activation Mapping is a technique to expose the  implicit attention
    of CNNs on the image. It highlights the most informative regions relevant
    to the predicted class

    Args:
        model: The pretrained GoogLeNet
        imagepaht: Path to the image for wicht to perform CAM
    """

    # transforms to prepare image for the net
    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # load the image
    img = Image.open(imagepath)
    img = img.convert('RGB')
    width, height = img.size

    # prepare it for model
    img = transformations(img)
    img = img.to(DEVICE)
    img.unsqueeze_(0)

    # prepare model
    model.eval()
    model.to(DEVICE)

    # get features of last conv layer for image
    features = model.get_last_conv(img)
    print(features.shape)

    # now perform average pooling to get weights for every feature
    avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
    weights = avg_pool(features)
    print(weights.shape)

    # convert to numpy and create heatmap
    features = features.to


model = GoogLeNet(10)
class_activation_mapping(model, "images/000877476.jpg")



