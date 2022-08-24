# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 20:36:43 2019

@author: luisf
"""
# =============================================================================
# DIP CLASS 6
#
#P2: Morphology: Erosion and Dilatation
#    Contour of a figure
# =============================================================================
 
import cv2
import os
import numpy as np

root = os.path.abspath(os.getcwd()) + '/images' 

#we load the image and transform it into grayscale to have only 1 layer
tri = cv2.imread(root + '/triangulo.tif') 
tri = cv2.cvtColor(tri, cv2.COLOR_BGR2GRAY)

#kernel element
kernel = np.ones((3,3) ,np.uint8)

#we will just erode the image, shrinking it a bit and then we subtract the
#the new shrinked to the original leaving only the contour of the figure
tri2 = cv2.erode(tri, kernel, iterations=1)
tri3 = cv2.subtract(tri,tri2)

#now we just sum the pixels of the contour (divided by 255 which because of the scale)
pixel_area = np.sum(tri3)/255

cv2.imshow('Contour ',tri3)

cv2.waitKey(0)
cv2.destroyAllWindows()

print('Area as pixels ',pixel_area)

