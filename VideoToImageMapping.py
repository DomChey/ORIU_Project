# -*- coding: utf-8 -*-
"""
Mapping from image filenames to videoframes folder

@author: Manuel
"""


# General mapping
import scipy.io
import numpy as np

matImage = scipy.io.loadmat('mpii_human_pose_v1_u12_1.mat')
matVideo = scipy.io.loadmat('mpii_human_pose_v1_sequences_keyframes.mat')

#print(matVideo['annolist_keyframes']['image'][0][2][0][0][0][0])

images = []
keyframes=[]
for i in range(24987):
    images.append(matImage['RELEASE']['annolist'][0][0][0][i][0][0][0][0][0])      
    keyframes.append(matVideo['annolist_keyframes']['image'][0][i][0][0][0][0])

mapping = []
for i,img in enumerate(images):
    mapping.append([img, keyframes[i]])
mapping = np.array(mapping, dtype=object)

        
# Mapping of the files we use for the classifier

file = open("final_images.txt", "r") 
finalImages = file.read().splitlines()
finalImages = np.array(finalImages, dtype=object)
    
finalMapping = []
for i,img in enumerate(finalImages):
    if img in mapping[:,0]:
        finalMapping.append([img, mapping[mapping[:,0]==img,1][0]])
    else:
        finalMapping.append([img, None])
finalMapping = np.array(finalMapping, dtype=object)
np.save("VideoToImageMapping", finalMapping)