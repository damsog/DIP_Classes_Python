# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 19:26:36 2019

@author: luisf
"""

# =============================================================================
# DIP with openCV Class 10
#
#P2: Finding the biggest Object using contours
# =============================================================================

# In the following code I will use findContours to find the objects on the
# number plate image that i have previously worked on, and find the biggest
# object to then delete all the other objects leaving only the number plate

import cv2
import numpy as np

root = 'D:/U de A/PDI/DIP_Clases_Python/'

# this function finds the biggest object on a binary image and deletes the
# smaller objects
def biggest_object(img):
    r,c,l = img.shape
    if( l > 1):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Thresholding. I should use the cv2 function but the image is already
    # binary in this case
    img[img > 0] = 255 
    
    # finding contours on the image
    _,contours,_ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # biggest  bounding Rectangle and area so far
    mx = (0,0,0,0)
    mx_area = 0
    
    # finding the biggest rectangle
    for cont in contours:
        # bounding rectangle for each contour and the area it takes
        x,y,w,h = cv2.boundingRect(cont)
        area = w*h
        if( area > mx_area):
            # new biggest area
            mx = x,y,w,h
            mx_area = area
    # removing the rest of the objects leaving only the biggest one
    x,y,w,h = mx
    img[0:y,:] = 0
    img[y+h:img.shape[0],:] = 0
    img[:,0:x] = 0
    img[:,x+w:img.shape[1]] = 0
    return mx_area, img        
    

plate = cv2.imread(root + 'images/placaCarro.tif')

area, nplate = biggest_object(plate)

cv2.imshow('Area: '+str(area),nplate)

cv2.waitKey(0)
cv2.destroyAllWindows()


