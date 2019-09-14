# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:19:22 2019

@author: luisf
"""

# =============================================================================
# DIP with openCV Class 13
#
#P2: Transformations 2: Geometric Transformations
#               Stretch, Shrink, Warp and Rotate
# =============================================================================

#------------------------------------------------------------------------------
# Affine Transform
# Here we will make some affine transformations to change the shape of a figure
# on an image. Affine transformations can be seen as having a paralelogram
# with vertices ABCD; applying an affine transformation becomes A'B'C'D'
# changing the paralelogram. Affine transformations are limited to 
# paralelograms but perspective transformations (Homography) allows to change a rectangle
# into a trapezoid

import cv2
import numpy as np

root = 'D:/U de A/PDI/DIP_Clases_Python/'
img = cv2.imread(root + 'images/imagen_2.jpg')

# Affine
# for this transformation we need to take 3 points of our original image
# and give them new positions and use a function to obtaine the transformation
# matrix

# 3 points of the original image. the points must be width,height not row,col
org = np.array([ [0,0],[img.shape[1]-1,0],[0,img.shape[0]-1] ]).astype(np.float32)
# 3 points as new positions
dst = np.array([ [0,img.shape[1]*0.15],[img.shape[1]*0.85,img.shape[1]*0.15],[img.shape[1]*0.15,img.shape[0]*0.7] ]).astype(np.float32)

# with the 2 triangles obtain the transformation matrix
mataff = cv2.getAffineTransform(org,dst)

# warp the image
img_warped = cv2.warpAffine(img, mataff, (img.shape[1], img.shape[0]))
cv2.imshow('Org Vs Warped', np.column_stack(( img,img_warped )) )

# Rotation
# to rotate the image using affine transform we have to generate the 
# transformation matrix

# center point of the rotation as x,y (not row,col)
center = (img.shape[1]//2, img.shape[0]//2)
angle = 45
scale = 1

#rotation transformation matrix
rot_mat = cv2.getRotationMatrix2D(center, angle, scale)

# warp the image
img_rotated = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
cv2.imshow('Org Vs Rotated', np.column_stack(( img,img_rotated )) )

#todo sparse


cv2.waitKey(0)
cv2.destroyAllWindows()