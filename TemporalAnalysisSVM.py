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
model_params = torch.load('175Epo.t7', map_location=lambda storage, loc: storage)
#GPU
#model_params = torch.load('175Epo.t7')

model = GoogLeNet(10)
model.load_state_dict(model_params['model'])

#Get labels and mapping
file = open("final_labels.txt", "r") 
labels = file.read().splitlines()
labels = np.array(labels, dtype=object)
mapping = np.load("VideoToImageMapping.npy")

#Get all features from the video frames
#X = []
#Y = []
#for i,m in enumerate(tqdm(mapping)):
#    #number of target frame
#    im_number = int(m[1][10:18])
#    
#    # Get 3, 5, 7 or 9 frames around target frame with 5 (4) frames spacing
#    features = []
#    for f in np.arange(im_number-16,im_number+20, 4):
#        path = str(f)
#        path = path.rjust(8,'0')
#        path = "E:/Video Data/"+m[1][0:10] + path + ".jpg"
#        
#        try:
#            im = prepareImage(path)
#            feat = model.get_features(im).detach().numpy()[0,:,0,0]
#            features.append(feat)
#        except:
#            print("error occured", sys.exc_info()[0])
#            break
#    
#    if len(features) > 0:
#        Y.append(int(labels[i]))
#        features = np.array(features).flatten()
#        X.append(features)
#        
#X, Y = np.array(X),np.array(Y)
#np.save("XData_SVM_9frames", X)

#Since this procedute is very time consuming you can run the line below instead of the loop above
#There a three pre-processed arrays:
#XData_SVM_3frames.npy : 3 frames around target frame with 5 frames spacing
#XData_SVM_5frames.npy : 5 frames around target frame with 5 frames spacing
#XData_SVM_7frames.npy : 7 frames around target frame with 5 frames spacing
#XData_SVM_9frames.npy : 9 frames around target frame with 4 frames spacing
X = np.load("XData_SVM_9frames.npy")
Y = labels

#Train Test split as used with the Neural Net
file = open("images/test_images.txt", "r") 
test_im = file.read().splitlines()
test_im = np.array(test_im, dtype=object)

file = open("images/train_images.txt", "r") 
train_im = file.read().splitlines()
train_im = np.array(train_im, dtype=object)

mask_test = []
mask_train = []

for im in train_im:
    idx = np.where(im==mapping[:,0])[0][0]
    mask_train.append(idx)
    
for im in test_im:
    idx = np.where(im==mapping[:,0])[0][0]
    mask_test.append(idx)
        
X_train = X[mask_train,:]
Y_train = Y[mask_train]
X_test = X[mask_test,:]
Y_test = Y[mask_test]

#Train the SVM
clf = SVC(kernel='linear')
clf.fit(X_train, Y_train) 
print("Training accuracy:", clf.score(X_train, Y_train))
print("Test accuracy", clf.score(X_test, Y_test))
        