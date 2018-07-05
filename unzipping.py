# -*- coding: utf-8 -*-
"""
Simple helper script for unzipping this mass of video data

@author: Manuel
"""
import tarfile

for i in range(12,16):
    print(" processing F:/mpii_human_pose_v1_sequences_batch"+ str(i) + ".tar.gz")
    tar = tarfile.open("F:/mpii_human_pose_v1_sequences_batch"+ str(i) + ".tar.gz")
    tar.extractall("F:/")
    tar.close()