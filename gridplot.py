# -*- coding: utf-8 -*-
"""
Simple script for plotting an iamge grid

@author: Manuel
"""

from PIL import Image
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt



data = open("final_images.txt", "r")

label = []
labels = open("final_labels.txt", "r")
for line in labels:
    label.append(int(line))

labels.close()
label = np.array(label)

pic_dim = 128

X = np.empty((label.shape[0],1,pic_dim,pic_dim))

for i,line in enumerate(data):
    image = Image.open("images/"+str(line[:-1]))
    width, height = image.size
    image = image.crop(((width-height)/2,0,width-(width-height)/2,height)) # Centre squared cropping
    image = image.resize((pic_dim,pic_dim), resample=Image.HAMMING)
    image = image.convert('L') #makes it greyscale
    matrix = np.asarray(image.getdata(),dtype=np.float)
    X[i,0,:,:] = np.reshape(matrix,(pic_dim,pic_dim))

data.close()



label = np.array(label)

print(len(label[label==0]), len(label[label==1]))



def plotgrid(X, n):
    dim = X.shape
    pic_dim = 128
    
    X = X.reshape((dim[0],pic_dim,pic_dim))
    
    grid = np.ones((n*pic_dim, n*pic_dim))
    
    for i in range(n):
        for j in range(n):
            grid[i*pic_dim:i*pic_dim+128, j*pic_dim:j*pic_dim+128] = X[i*n+j,:,:]
            
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(grid, cmap='gray')
    fig.savefig("plotgrid.png", dpi=300)

    plt.show()
    
X = X[np.random.permutation(1576),:,:,:]

plotgrid(X,20)