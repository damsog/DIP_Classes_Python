# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 21:06:27 2019

@author: luisf
"""

# =============================================================================
# DIP CLASS 6
#
#P3: Morphology: Closing and Opening
#    
# =============================================================================

#If we dilate an image some holes will be filled and then we erode to go back
#to the normal size but some holes got filled. this is know as closing
#on the other hand. if we erode first and then we dilate small dots and thin 
#regions will be erased. this is known as opening
#
#Applying these operations we can delete thrash or we can fill gaps
#

import cv2
import os
import numpy as np
#this function takes an image and sets to 0 a porcentage of the border
from MyClearBorder.ClearBorder import myclearborder

root = os.path.abspath(os.getcwd()) + '/images' 

#lets load the image of the car
placaColor = cv2.imread(root + 'images/carro.jpg')
r,c,l = placaColor.shape

#and lets load the image of the number plate that we have been processing
placa = cv2.imread(root + 'images/placaCarro.tif')

#changing its space to grayscale and setting to 255 anything that is not 0
#this will be our mask to cut the number plate on the car image
placa = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
placa[placa>0]=255
cv2.imshow('Placa 1',placa)

#making a big kernel. the bigger the kernel the dilation and erosion 
#will have a bigger impact (similar to do more iterations)
kernel = np.ones((30,30) ,np.uint8)
#this second kernel is to make a test
kernel2 = np.ones((8,8) ,np.uint8)

#We want to remove the gaps in our number plate
#so the best to do is closing and then we open to go back to normal size
placa1 = cv2.morphologyEx(placa, cv2.MORPH_CLOSE, kernel)
placa3 = cv2.morphologyEx(placa1, cv2.MORPH_OPEN, kernel)

#This is just a test to compare closing then opening to opening then closing
placa2 = cv2.morphologyEx(placa, cv2.MORPH_OPEN, kernel2)
placa4 = cv2.morphologyEx(placa2, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Close->Open',placa3)
#cv2.imshow('Open->Close',placa4)

#with the mask processed (closed->opened) we need to clear the thrash around
#so lets clear its borders 20%
placa3 = myclearborder(placa3, 0.2,0.2)
cv2.imshow('Cleared ',placa3)

#resizing to the lengths of the original image
placa3 = cv2.resize(placa3, ( c,r ) )

#finding the coordenates of the mask
r3,c3=np.where(placa3>0)[0],np.where(placa3>0)[1]
rmin = np.min(r3)
rmax = np.max(r3)
cmin = np.min(c3)
cmax = np.max(c3)

#cutting the original image to the coordenates of the mask
placaColor = placaColor[ rmin:rmax, cmin:cmax ,: ]

#the window kept appearing outside the screen so i had to move it
cv2.namedWindow('Number Plate Sectioned')
cv2.moveWindow('Number Plate Sectioned', 400,250)
cv2.imshow('Number Plate Sectioned',placaColor)

cv2.imwrite(root + 'images/placa_p.jpg',placaColor)

cv2.waitKey(0)
cv2.destroyAllWindows()

    