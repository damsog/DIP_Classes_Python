# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 22:55:37 2019

@author: luisf
"""

# =============================================================================
# DIP CLASS 9
#
#P3: Distance Transformation
#    
# =============================================================================

# A distance transformation is an operator that calculates and assigns the
# distance of each pixel (bright) to the nearest zero valued pixel, creating 
# a new image composed of zero values and distances for bright pixels

import cv2
import numpy as np

root = 'D:/U de A/PDI/DIP_Clases_Python/'

# Reading the image and converting it to grayscale
img = cv2.imread(root + 'images/figuras_3.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# applying a threshold. this transformation works on binary images
_,img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
cv2.imshow('Original',img)

# Applying the distance transformation, calculating the euclidean distance (L2)
img2 = cv2.distanceTransform(img, cv2.DIST_L2, 3)

# normalizing the values of the new image for 0 - 1
cv2.normalize(img2, img2, 0, 1, cv2.NORM_MINMAX)

# showing the result
cv2.imshow('Disntance Transform',img2)

# each line is the distance to the nearest zero valued pixel

cv2.waitKey(0)
cv2.destroyAllWindows()