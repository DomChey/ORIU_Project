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

np.random.seed(10)

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
#CPU
#model_params = torch.load('175Epo.t7', map_location=lambda storage, loc: storage)
#GPU
model_params = torch.load('175Epo.t7')

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
    
    # Get 3, 5 or 7 frames around target frame with 5 frames spacing
    features = []
    for f in np.arange(im_number-15,im_number+20, 5):
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
        
X, Y = np.array(X),np.array(Y)
np.save("XData_SVM_7frames", X)

#Since this procedute is very time consuming you can run the line below instead of the loop above
#X = np.load("XData_SVM.npy")


#Train Test split at 80% with random shuffle
dim = Y.shape[0]
shuffle = np.random.permutation(dim)
X = X[shuffle,:]
Y = Y[shuffle]
split = int(np.round(dim*0.8))
X_train = X[0:split,:]
Y_train = Y[0:split]
X_test = X[split::,:]
Y_test = Y[split::]

#Train the SVM
clf = SVC(kernel='linear')
clf.fit(X_train, Y_train) 
print("Training accuracy:", clf.score(X_train, Y_train))
print("Test accuracy", clf.score(X_test, Y_test))
        