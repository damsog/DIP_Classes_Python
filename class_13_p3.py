# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 21:54:24 2019

@author: luisf
"""


# =============================================================================
# DIP with openCV Class 13
#
#P3: Transformations 2: Geometric Transformations
#               Perspective Transformation
# =============================================================================

#------------------------------------------------------------------------------
# Perspective transforms (Homography)
# perspective transformations are more flexible than affine where the points
# have to keep a parallelogram form while using perspective transformations
# allows us to change rectangles into trapezoids and vice versa, which lets us
# change the perspective of an object on the image

import cv2
import os
import numpy as np

root = os.path.abspath(os.getcwd()) + '/images' 
img = cv2.imread(root + 'images/imagen_2.jpg')

# Similar to affine transformation we have to create a transformation matrix
# but this time has to be 3x3 instead of 3x2
# opencv can create it for us using getPerspectiveTransform, we only have to
# indicate the 4 points of origin and the 4 points for the new perspective

pts_org = np.array([ [0,0],
                    [img.shape[1],0],
                    [img.shape[1],img.shape[0]],
                    [0,img.shape[0]] ]).astype(np.float32)
pts_dst = np.array([ [0,0],
                    [img.shape[1]*0.75,img.shape[0]*0.15],
                    [img.shape[1]*0.75,img.shape[0]*0.75],
                    [0,img.shape[0]] ]).astype(np.float32)
tmatrix = cv2.getPerspectiveTransform(pts_org, pts_dst)

img_pers = cv2.warpPerspective(img, tmatrix, (img.shape[1],img.shape[0]))

cv2.imshow("ORG vs Perspective", np.column_stack(( img,img_pers )) )

# Todo sparse

cv2.waitKey(0)
cv2.destroyAllWindows()