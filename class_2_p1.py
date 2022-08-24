# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 11:57:17 2019

@author: luisf
"""

# =============================================================================
# DIP with openCV Class 2
#
#Showing images and reshaping
# =============================================================================

import cv2
import os
import numpy as np
import os

root = os.path.abspath(os.getcwd()) + '/images' 

a = cv2.imread(root + '/paisaje.jpg')  #Reading an image
b = cv2.imread(root + '/carro.jpg')
#a color image is composed of 3 arrays (layers) each one with size rowsXcolumns
#each row x column x layer (which is the color) has a value from 0 to 255
#combining the values of each color (RGB) on a rowXcolumn we obtain a certain pixel

#getting row, column and layer size
rows1,cols1,layers1 = a.shape     
rows2,cols2,layers2 = b.shape

#cutting b with the sizes of a
b1=b[0:rows1,0:cols1,0:3]
#to show the 2 in one image with openCV
#we have to stack them horizontally (stack the columns)
ab1 = np.concatenate((a,b1),axis=1)
cv2.imshow('frame1',ab1)

#resize uses width, heigh wo in this case
#width=cols and height=rows
b2=cv2.resize(b,(cols1,rows1),interpolation = cv2.INTER_AREA)

ab2=np.concatenate((a,b2),axis=1)
cv2.imshow('frame2',ab2)

#waiting indefinetly till a key is pressed and destroy all windows at the end
cv2.waitKey(0)
cv2.destroyAllWindows()



