# -*- coding: utf-8 -*-
"""
Search the video data to determine if all neccesary sequences are available

@author: Manuel
"""

import numpy as np
import os

#Scan all folders in F:/ (where all video data is saved)

dirs = []
for root, d, f in os.walk("F:/"):
   for name in d:
      dirs.append(name)

mapping = np.load("VideoToImageMapping.npy")

available = []
for vid in mapping[:,1]:
    if vid[0:9] in dirs:
        available.append(1)
    else:
        available.append(0)
available = np.array(available)

print("{:.4f}% is available".format(100*np.sum(available)/available.shape[0]))