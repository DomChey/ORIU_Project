# -*- coding: utf-8 -*-
"""
SVM algorithm for temporal analysis of the video data

@author: Manuel
"""

import torch
from torch.utils.serialization import load_lua
import numpy as np
from PIL import Image
from googLeNet import GoogLeNet

#initialize the model and training data
model_params = torch.load('40EpoStandard.t7')
model = GoogLeNet(10)
model.load_state_dict(model_params['model'])
 
im = Image.open("images/000003072.jpg")
pix = np.array(im)
im_tensor = torch.from_numpy(pix)

feat = model.get_features(im_tensor)
print(feat)