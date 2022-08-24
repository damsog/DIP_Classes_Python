# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 01:25:39 2019

@author: luisf
"""

# =============================================================================
# DIP with openCV Class 16
#
#P2:    Textures 1
#       Statistical Segmentation
# =============================================================================

# For Texture analysis we want to get information about a region on an image.
# this information or features can be statistical information like the mean
# and the std deviation or the variance


import cv2
import os
import numpy as np

# this function takes an image and gets some statistical information about
# a small window 10x10 and store it on another matrix and then move the window.
# we end up with a new image of means and another of deviations
def statSegm(img):
    r,c = img.shape

    r_ini = 5
    c_ini = 5
    r_end = r - 5
    c_end = c - 5
    
    # the new image is 10x10 smaller than the original
    img_med = np.zeros((r-5,c-5), np.float32)
    img_std = np.zeros((r-5,c-5), np.float32)
    
    
    rows = 0
    
    #passing the window through all the image
    for i in range(r_ini, r_end ):
        cols = 0
        for j in range(c_ini, c_end):
            window = img[ i-5:i+5 , j-5:j+5 ]
            
            # calculate the mean and std of the window and store it as a pixel
            img_med[rows,cols],img_std[rows,cols] = cv2.meanStdDev(window)
            cols = cols + 1
        
        rows = rows + 1
    
    #normalizing, rounding and casting to uint8 which is the format for images
    img_med = np.around( img_med*255 / (np.max(img_med)) ).astype(np.uint8)
    img_std = np.around( img_std*255 / (np.max(img_std)) ).astype(np.uint8)
    
    return img_med, img_std
    

root = os.path.abspath(os.getcwd()) + '/images' 
img = cv2.imread(root + '/river.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
r,c = img.shape
#img = cv2.resize(img , ( int(c*0.7), int(r*0.7) ), interpolation = cv2.INTER_AREA)

img_med, img_std = statSegm(img)

# the first mean and std are useful for segmenting different textures
cv2.imshow('ORG', img)
cv2.imshow('Med', img_med)
cv2.imshow('Std', img_std)

# using the mean or std as input to calculate again the mean and std 
# the new std shows the regions with different textures and
# highlights the borders of the textures.

img_med2, img_std2 = statSegm(img_std)

cv2.imshow('Med2', img_med)
cv2.imshow('Std2', img_std)

# doing this again, mean or std as input, the new std shows only the borders
# where the textures touch

img_med3, img_std3 = statSegm(img_med)


cv2.imshow('Med3', img_med3)
cv2.imshow('Std3', img_std3)

img2 = img[0:10,0:10]

cv2.waitKey(0)
cv2.destroyAllWindows()