# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:59:28 2019

@author: luisf
"""

# =============================================================================
# DIP with openCV Class 12
#
#P3: Transformations 1: Detectors
#               
# =============================================================================

#------------------------------------------------------------------------------
# Understood the filters and derivatives operators from the previous section
# we can now understand some useful detectors operators for finding edges
# and different geometrical forms 

import cv2
import os
import numpy as np

root = os.path.abspath(os.getcwd()) + '/images' 
img = cv2.imread(root + '/imagen_2.jpg')
img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#------------------------------------------------------------------------------
#------------------------Canny edge detector-----------------------------------
# This operator uses derivatives similar to sobel to find edges on an image
# the procedure roughly consists of a first step for filtering the image
# then finding the derivatives of the image and their direction
# then zones that are not part of edges are suppressed; and for last
# two thresholds are applied, if a pixel is below the lower one then is
# suppressed, if its above the high threshold then is spared. but if its
# in between the thresholds is spared only if its connected to pixel above the
# higher one.

lower_threshold = 240
higher_threshold = 250

# Using canny for a kernel size of 3x3
img_edges = cv2.Canny(img_g, lower_threshold, higher_threshold, 3 )

cv2.imshow('ORG vs Edges', np.column_stack((img_g,img_edges)) )

#------------------------------------------------------------------------------
#------------------------Hough Transforms--------------------------------------
# Hough transform are operators that allow to detect certain simple forms

# Hough Line transform
# Finds the lines on an image and represents them in polar form
# r = x*cost + y*sint
img2 = cv2.imread(root + '/simpsons_house.jpg')
img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)



cv2.waitKey(0)
cv2.destroyAllWindows()