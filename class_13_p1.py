# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 20:15:24 2019

@author: luisf
"""

# =============================================================================
# DIP with openCV Class 13
#
#P1: Transformations 2: Remapping
#               
# =============================================================================

#------------------------------------------------------------------------------
# Remapping is taking the pixels on an image and changing their position
# to a new in order to change de form of an image in a way that it keeps
# like flipping it, reflecting it, resizing it. etc.

import cv2
import numpy as np

root = 'D:/U de A/PDI/DIP_Clases_Python/'
img = cv2.imread(root + 'images/imagen_2.jpg')

# reflecting an image

# we need to create the map matrices for x and y directions
# these need to have the same size as the image
mapx = np.zeros((img.shape[0],img.shape[1]), np.float32)
mapy = np.zeros((img.shape[0],img.shape[1]), np.float32)

# for reflecting x
#a matrix for x in which each row containes the indices of the columns 
#(x axis) in descendence, to get the img inverted on x
for i in range(img.shape[0]):
    mapx[i,:] = [img.shape[1]-x for x in range(img.shape[1])]
#we dont want y inverted. each column containes the indices of the rows 
#(y axis) in ascendence
for j in range(img.shape[1]):
    mapy[:,j] = [y for y in range(img.shape[0])]

img_reflx = cv2.remap(img, mapx, mapy, cv2.INTER_AREA)

# for reflecting y
#a matrix for x in which each row containes the indices of the columns 
#(x axis) in ascendance
for i in range(img.shape[0]):
    mapx[i,:] = [x for x in range(img.shape[1])]
#we dont want y inverted. each column containes the indices of the rows 
#(y axis) in descendence
for j in range(img.shape[1]):
    mapy[:,j] = [img.shape[0]-y for y in range(img.shape[0])]

img_refly = cv2.remap(img, mapx, mapy, cv2.INTER_AREA)

# for reflecting both
#a matrix for x in which each row containes the indices of the columns 
#(x axis) in descendence
for i in range(img.shape[0]):
    mapx[i,:] = [img.shape[1]-x for x in range(img.shape[1])]
#we dont want y inverted. each column containes the indices of the rows 
#(y axis) in descendence
for j in range(img.shape[1]):
    mapy[:,j] = [img.shape[0]-y for y in range(img.shape[0])]
    
img_reflxy = cv2.remap(img, mapx, mapy, cv2.INTER_AREA)


cv2.imshow('ORG vs Reflxy || Reflx vs Refly', np.row_stack(( np.column_stack(( img,img_reflxy )),np.column_stack(( img_reflx,img_refly )) )) )

# for resizing the imag with remapping. lest reduce it to 
#we have to iterate for allthe pixels
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        #we have to make 0 the pixels outside our new size. Anthe pixels inside
        #we have to remap them (!)
        if( (i > img.shape[0]*0.25) and (i < img.shape[0]*0.75) and (j > img.shape[1]*0.25) and (j < img.shape[1]*0.75) ):
            mapx[i,j] = 2*(j - img.shape[1]*0.25) + 0.5
            mapy[i,j] = 2*(i - img.shape[0]*0.25) + 0.5
        else:
            mapx[i,j] = 0
            mapy[i,j] = 0


img_res = cv2.remap(img, mapx, mapy, cv2.INTER_AREA)

cv2.imshow('ORG vs Resized', np.column_stack(( img,img_res )) )

cv2.waitKey(0)
cv2.destroyAllWindows()


