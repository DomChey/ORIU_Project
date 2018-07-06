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
import numpy as np
from matplotlib import cm


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
    orig_img = Image.open(imagepath)
    orig_img = orig_img.convert('RGB')
    width, height = orig_img.size

    # prepare it for model
    img = transformations(orig_img)
    img = img.to(DEVICE)
    img.unsqueeze_(0)

    # prepare model
    model.eval()
    model.to(DEVICE)

    # get features of last conv layer for image
    features = model.get_last_conv(img)
    # and get output of model 
    prediction = model(img)
    # softmax to geht class prediction
    softmax = F.softmax(prediction, dim=1)
    # index of most likely class
    _, idx = softmax.max(1)
    # get the weights of the last layer
    weights = model.last_linear.weight
    # only keep the weights for the predicted class
    weights = weights[idx]

    # convert to numpy
    features = features.to(torch.device("cpu"))
    features = features.squeeze()
    features = features.detach().numpy()
    weights = weights.to(torch.device("cpu"))
    weights = weights.squeeze()
    weights = weights.detach().numpy()

    # create heatmap
    heatmap = np.zeros((7, 7))
    for i in range(1024):
        heatmap = heatmap + features[i] * weights[i]
    # normalize the heatmap
    heatmap = heatmap / np.max(heatmap)
    # save it as colorimage
    cm_hot = cm.hot
    heatmap = cm_hot(heatmap)
    im = Image.fromarray(np.uint8(heatmap * 225))
    im = im.resize((width, height), Image.ANTIALIAS)
    im = im.convert('RGB')
    im.save('test.jpg')


def load_model(model):
    checkpoint = torch.load('ckpt.t7')
    model.load_state_dict(checkpoint['model'])
    return model


model = GoogLeNet(10)
model = load_model(model)
class_activation_mapping(model, "images/000111209.jpg")
