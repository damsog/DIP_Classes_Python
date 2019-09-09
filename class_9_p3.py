# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 17:32:43 2019

@author: luisf
"""

# =============================================================================
# DIP CLASS 9
#
#P3: Morphology: More morpholy operations
#     Floodfill
# =============================================================================

# Floodfill is an operation that starts fromapoint (given) and every pixel
# it finds that is 0 sets it to max. in other words, fills the background of 
# an image with a value that we want

import cv2
import numpy as np

root = 'D:/U de A/PDI/DIP_Clases_Python/'

img = cv2.imread('images/tofill.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
r,c = img.shape
nh = int(r*1)
nw = int(c*1)
img2 = cv2.resize(img, ( nw,nh ), interpolation=cv2.INTER_AREA)
h, w = img2.shape

#we have to create a mask that is 1 pixel bigger than our image to fill in 
#every direction
mask = np.zeros(( h + 2 , w + 2 ) , np.uint8)

# this point is where we will start tofill
point = (0,0) 
img_tofill = img2.copy()
#filling with a value of 100
cv2.floodFill(img_tofill,mask, (0,0), 100)

cv2.imshow('Img', np.column_stack((img2,img_tofill )) )

mask = mask*0 #setting themask back to 0
#lets use this function to remove holes in a binary image
point = (0,0) 
img_tofill = img2.copy()
#filling with a value of 255
cv2.floodFill(img_tofill,mask, (0,0), 255)


#now because the figures have a value of 255, the background
#has merged with the figures leaving only the holes

#now lets invert the image so the holes are set to 255 and the rest is 0
img_tofill = cv2.bitwise_not(img_tofill)


#and finally lets apply an OR gate between the filled image and our normal img
img_tofill = img2 | img_tofill

cv2.imshow('Result', np.column_stack((img2, img_tofill)) )




cv2.waitKey(0)
cv2.destroyAllWindows()
