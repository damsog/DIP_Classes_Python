# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 21:07:27 2019

@author: luisf
"""

# =============================================================================
# DIP with openCV Class 12
#
#P2: Transformations 1: Convolution
#               Gradients
# =============================================================================

#------------------------------------------------------------------------------
# In this part we will take a look into derivatives of images 
# The simplest approximation of derivatives on images ir the sobel derivative
# Since we are dealing with 2D objects we have to specify the order of the
# derivative on a certain direction

import cv2
import os
import numpy as np

root = os.path.abspath(os.getcwd()) + '/images' 
img = cv2.imread(root + '/imagen_2.jpg')
img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# here we are derivating on x using a kernel of size 5x5
dimgdx = cv2.Sobel(img_g,-1, dx=1,dy=0, ksize=3)
# here we are derivating on y using a kernel of size 5x5
dimgdy = cv2.Sobel(img_g,-1, dx=0,dy=1, ksize=3)
# here we are derivating both on x and y
dimgdxdy = cv2.Sobel(img_g,-1, dx=1,dy=1, ksize=3)

# lets stack the org and the derivatives on a single image
stack = np.column_stack(( img_g,dimgdxdy ))
stack = np.row_stack(( stack,np.column_stack(( dimgdx,dimgdy )) ))
cv2.imshow('ORG vs DxDy | Dx vs Dy', stack)

#------------------------------------------------------------------------------
# Scharr filter is a modification of the sobel approximation
# that is useful for cases when sobel is not very acurate, for small kernels

# here we are derivating on x using a kernel of size 5x5
dimgdx_s = cv2.Scharr(img_g,-1, dx=1,dy=0)
# here we are derivating on y using a kernel of size 5x5
dimgdy_s = cv2.Scharr(img_g,-1, dx=0,dy=1)
# here we are derivating both on x and y

#Scharr filter works only when dx+dy==1.

# lets stack the org and the derivatives on a single image
stack2 = np.column_stack(( dimgdy_s,dimgdx_s ))

cv2.imshow('Using Schar Dx vs Dy', stack2)

#------------------------------------------------------------------------------
# Laplacian Derivative
# The laplace operator describes the second spatial derivatives
# L(f) = d2f/dx2 + d2f/dy2 
# The computational form of this operator is described with the sobel
# operator for second order derivatives

# here we are derivating on x using a kernel of size 5x5
d2imgdx2 = cv2.Sobel(img_g,-1, dx=2,dy=0, ksize=3)
# here we are derivating on y using a kernel of size 5x5
d2imgdy2 = cv2.Sobel(img_g,-1, dx=0,dy=2, ksize=3)
# here we are derivating both on x and y
d2imgdx2dy2 = cv2.Sobel(img_g,-1, dx=2,dy=2, ksize=3)

# lets stack the org and the derivatives on a single image
stack3 = np.column_stack(( img_g,d2imgdx2dy2 ))
stack3 = np.row_stack(( stack3,np.column_stack(( d2imgdx2,d2imgdy2 )) ))
cv2.imshow('ORG vs D2xD2y | D2x vs D2y', stack3)

# Applying the laplacian operator
Limg = cv2.Laplacian(img_g, -1, 3)

# the Laplacian operator on images is useful to detect blobs on an image
cv2.imshow('D2xD2y vs L', np.column_stack(( d2imgdx2dy2,Limg )) )

cv2.waitKey(0)
cv2.destroyAllWindows()
