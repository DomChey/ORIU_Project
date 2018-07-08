# -*- coding: utf-8 -*-
"""
SVM algorithm for temporal analysis of the video data

@author: Manuel
"""

import torch
from torch.utils.serialization import load_lua
from torchvision import transforms
import numpy as np
from PIL import Image
from googLeNet import GoogLeNet
import sys
from tqdm import tqdm
from sklearn.svm import SVC


def prepareImage(path):
    #Function that loads an image and converts it to a useable Torch tensor:
    #resizing, correct dimension order and conversion to float tensor
    im = Image.open(path)
    im = im.resize((224, 224))
    pix = np.array(im)
    pix = np.swapaxes(pix, 1,2)
    pix = np.swapaxes(pix, 0,1)
    im_tensor = torch.from_numpy(pix)
    im_tensor = im_tensor.type(torch.FloatTensor)
    im_tensor.unsqueeze_(0)
    return im_tensor
    

#initialize the model and training data
model_params = torch.load('93Epo5Crop.t7')
model = GoogLeNet(10)
model.load_state_dict(model_params['model'])

#Get labels and mapping
file = open("final_labels.txt", "r") 
labels = file.read().splitlines()
labels = np.array(labels, dtype=object)
mapping = np.load("VideoToImageMapping.npy")

#Get all features from the video frames
X = []
Y = []
for i,m in enumerate(tqdm(mapping)):
    #number of target frame
    im_number = int(m[1][10:18])
    
    # Get 5 frames around target frame with 5 frames spacing
    features = []
    for f in np.arange(im_number-10,im_number+15, 5):
        path = str(f)
        path = path.rjust(8,'0')
        path = "F:/Video Data/"+m[1][0:10] + path + ".jpg"
        
        try:
            im = prepareImage(path)
            feat = model.get_features(im).detach().numpy()[0,:,0,0]
            features.append(feat)
        except:
            print("error occured", sys.exc_info()[0])
            break
    
    if len(features) > 0:
        Y.append(int(labels[i]))
        features = np.array(features).flatten()
        X.append(features)
    
#newX = []
#newY = []
#for i,x in enumerate(X):
#    if len(x)>0:
#        newX.append(x)
#        newY.append(int(labels[i]))
#        
#newX, newY = np.array(newX),np.array(newY)

#Train the SVM
clf = SVC(kernel='linear')
clf.fit(newX, newY) 
print(clf.score(newX, newY))
        