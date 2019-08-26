# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 22:52:09 2019

@author: luisf
"""

# =============================================================================
# DIP CLASS 9
#
#P2: Morphology: More morpholy operations
#     Top-Hat, Black-Hat, Gradient, Flood Fill*
# =============================================================================

# There are more morphological operations based on the erosion and dilation

import cv2
import numpy as np
root = 'D:/U de A/PDI/DIP_Clases_Python/'

img = cv2.imread('images/carro(1).jpg')
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
r,c = imggray.shape
imggray = cv2.resize(imggray, (int(c*0.2),int(r*0.2)), interpolation=cv2.INTER_AREA)

kernel = np.ones((3,3), np.uint8)

# the morphological gradient determines how fast brigthness is changing on
# an image and its defined by G = Dilate(img) - Erode(img).

#img_grad = cv2.dilate(imggray, kernel)
#img_grad = img_grad - cv2.erode(imggray, kernel)

img_grad = cv2.morphologyEx(imggray, cv2.MORPH_GRADIENT, kernel)

cv2.imshow('Morphological Gradient', np.column_stack((imggray,img_grad)) )

# the next operations use opening and closing. recall that these operations
# are eroding->dilating and dilating->eroding respectibly

# Next operator is the top-hat. This operation isolate patches that are 
# brighter than its surroundings
# top-hat(img) = img - open(img)

img_tophat = cv2.morphologyEx(imggray, cv2.MORPH_TOPHAT, kernel)

cv2.imshow('Top-Hat', np.column_stack((imggray,img_tophat)) )

# Next operator is the black-hat. This operation isolate patches that are 
# darker than its surroundings
# black-hat(img) = closing(img) - img

img_blackhat = cv2.morphologyEx(imggray, cv2.MORPH_BLACKHAT, kernel)

cv2.imshow('Black-Hat', np.column_stack((imggray,img_blackhat)) )

# Pending FloodFill

cv2.waitKey(0)
cv2.destroyAllWindows()



