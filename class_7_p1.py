# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 15:46:10 2019

@author: luisf
"""

# =============================================================================
# DIP CLASS 7
#
#P1: Basic Histogram
#    
# =============================================================================

# An histogram is the count of pixels for each value of intensity 
# (0-255 in this case). in other words, is the distribution of intensity of the
# image.
# Histogram Equalization is a process that increases the contrast stretching 
# the histogram, increasing the number of low populated intensities and decreasing
# high populated intensities

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

root = os.path.abspath(os.getcwd()) + '/images' 

carro = cv2.imread(root + '/carro(1).jpg')
r,c,l = carro.shape
carro = cv2.resize( carro, ( int(c*0.2),int(r*0.2)) , interpolation=cv2.INTER_AREA )
carroG = cv2.cvtColor(carro, cv2.COLOR_BGR2GRAY)
#histCarro = cv2.calcHist([carroG],[0],None,[256],[0,256])

# type the following to show images on a new window on spyder 
# %matplotlib qt 
# and type the following to get them back on the terminal
# %matplotlib inline

#plotting the image vs its histogram
fig, [ax1,ax2] = plt.subplots(2 , 1)
fig.suptitle('Car Vs Histogram')
ax1.axis('off')
ax1.imshow(carroG)
ax2.hist(carroG.ravel(),256,[0,256])

#applying histogram equalization
carroG_equalized = cv2.equalizeHist(carroG)

#showing before and after equalizacion using subplots
fig2, ax= plt.subplots(2 , 2)
fig.suptitle('Car Vs Histogram')
ax[0,0].imshow(carroG)
ax[0,0].set_title('Original')
ax[0,1].hist(carroG.ravel(),256,[0,256])
ax[0,1].set_title('Histogram')
ax[1,0].imshow(carroG_equalized)
ax[1,0].set_title('Equalized')
ax[1,1].hist(carroG_equalized.ravel(),256,[0,256])
ax[1,1].set_title('Histrogram Equalized')

#Showing the grayscale images before and after equalization
cv2.imshow('Before After Histogram Equalization', np.column_stack((carroG,carroG_equalized)) )

cv2.waitKey(0)
cv2.destroyAllWindows()