# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 14:10:19 2019

@author: luisf
"""
# =============================================================================
# DIP with openCV Class 2
#
#P2: RGB Components
# =============================================================================

import cv2
import numpy as np

root = 'D:/U de A/PDI/DIP_Clases_Python/'

a = cv2.imread(root + 'images/paisaje.jpg')  #Reading an image
cv2.imshow('frame1',a)         #showing original image

#We want to decompose a on its rgb components
#so we create a new copy of a for each color and
#we will set the other two to 0 remaining onlly one component

#NOTE: REMEMBER that assigning a variable to another 
#DOES NOT create a copy on PYTHON (most languages do, but python doesn't for some reason)
#so we will have to use a function that copies
a_red = np.copy(a)
a_green = np.copy(a)
a_blue = np.copy(a)

#Setting components to 0 remaining only one for each case
#to see them separately
a_blue[:,:,1:3]=0
a_green[:,:,0:1]=0
a_green[:,:,2]=0
a_red[:,:,0:2]=0

#concatenating them horizontally to show them on one frame
a_rgb = np.concatenate((a_blue,a_green), axis=1)
a_rgb = np.concatenate((a_rgb,a_red), axis=1)

cv2.imshow('frame2',a_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()