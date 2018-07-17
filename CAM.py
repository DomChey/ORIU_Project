"""
Implementation of Class Activatoin Mapping
@author: Dominique Cheray
"""

# necessary imports
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet
from googLeNet import GoogLeNet
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib import cm


# usefull globals
USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'


def class_activation_mapping(model, imagepath, isGoogLeNet):
    """
    Class Activation Mapping is a technique to expose the  implicit attention
    of CNNs on the image. It highlights the most informative regions relevant
    to the predicted class

    Args:
        model: The pretrained GoogLeNet
        imagepath: Path to the image on wich to perform CAM
        isGoogLeNet: If the given model is our implementation of the GoogLeNet or not

    """

    # class mapping
    classes = ["basketball", "horseback riding", "martial arts", "paddleball", "rock climbing", "rope skipping", "skateboarding", "softball", "tennis", "golf"]

    # get name of image for later use
    name = imagepath.split("/")[1]

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
    if isGoogLeNet:
        features = model.get_last_conv(img)
    else:
        layer = model._modules.get('layer4')
        features = torch.zeros(1, 512, 7, 7)
        def copy_features(model, i, o):
            features.copy_(o.data)
        hook = layer.register_forward_hook(copy_features)

    # and get output of model
    prediction = model(img)
    if not isGoogLeNet:
        hook.remove()
    # softmax to geht class prediction
    softmax = F.softmax(prediction, dim=1)
    # index of most likely class
    _, idx = softmax.max(1)
    # get the weights of the last layer
    if isGoogLeNet:
        weights = model.last_linear.weight
    else:
        weights = model.fc.weight
    # only keep the weights for the predicted class
    weights = weights[idx]

    # convert to numpy
    idx = idx.cpu().numpy()[0]
    features = features.to(torch.device("cpu"))
    features = features.squeeze()
    features = features.detach().numpy()
    weights = weights.to(torch.device("cpu"))
    weights = weights.squeeze()
    weights = weights.detach().numpy()

    # create heatmap
    heatmap = np.zeros((7, 7))
    if isGoogLeNet:
        for i in range(1024):
            heatmap = heatmap + features[i] * weights[i]
    else:
        for i in range(512):
            heatmap = heatmap + features[i] * weights[i]
    # normalize the heatmap
    heatmap = heatmap / np.max(heatmap)
    # save it as colorimage
    colormap = cm.hot
    heatmap = colormap(heatmap)
    heatmap = Image.fromarray(np.uint8(heatmap * 225))
    heatmap = heatmap.resize((width, height), Image.ANTIALIAS)
    # make the heatmap transparent
    heatmap.putalpha(200)
    # now fuse image and heatmap together
    orig_img.paste(heatmap, (0, 0), heatmap)
    # write class prediction into image
    drawable = ImageDraw.Draw(orig_img)
    font = ImageFont.truetype('Roboto-Bold.ttf', 20)
    drawable.text((10, 10), classes[idx], font=font, fill=(255, 255, 255))
    # save the image
    orig_img.save("CAMs/heatmap_{}".format(name))


def load_model(model, checkpoint):
    """
    Method to load a pretrained network

    Args:
       model: Network to add the loaded parameters to"""
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model'])
    return model


# load the trained model
model = GoogLeNet(10)
#model = resnet.resnet34(pretrained=False, num_classes=10)
model = load_model(model, 'GoogLeNet.t7')
# get all test images
test_images_file = open("images/test_images.txt")
test_images = test_images_file.read().splitlines()
test_images_file.close()
# create a CAM image for all test images
for image in test_images:
    imagepath = "images/" + image
    class_activation_mapping(model, imagepath, True)
