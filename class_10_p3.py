# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:53:23 2019

@author: luisf
"""

# =============================================================================
# DIP with openCV Class 10
#
#P3: Skeletonization 
#
#   Hilditch's Algorithm
#       for
#   Skeletonization
# =============================================================================

import cv2
import os
import numpy as np

root = os.path.abspath(os.getcwd()) + '/images' 

fig = cv2.imread('images/figura2.bmp')
fig = cv2.cvtColor(fig, cv2.COLOR_BGR2GRAY)
size = np.size(fig)

# a black image with the size of the original
skel = np.zeros(fig.shape , np.uint8)

ret, fig = cv2.threshold(fig, 127, 255,0)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

finished = False

# temp = fig - opened(fig)
# skel = temp | fig
# fig = erode(fig)
while(not finished):
    eroded = cv2.erode(fig, kernel)
    temp = cv2.dilate(eroded, kernel)
    temp = cv2.subtract(fig, temp)
    skel = cv2.bitwise_or(skel, temp)
    fig = eroded.copy()
    
    # counts how many zeros on the image
    # when the image has been eroded to only zeros then stops
    zeros = size - cv2.countNonZero(fig)
    if(zeros == size):
        finished = True

cv2.imshow('Skel ',skel)
cv2.waitKey(0)
cv2.destroyAllWindows()
