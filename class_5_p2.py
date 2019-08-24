# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 17:44:48 2019

@author: luisf
"""

# =============================================================================
# DIP CLASS 5
#
#P2: Using Color Layers to separate the license plate on an image
#    using Faster method
# =============================================================================

import cv2
import numpy as np
root = 'D:/U de A/PDI/DIP_Clases_Python/'
carro = cv2.imread(root + 'images/carro.jpg')

r,c,l = carro.shape

#resizing because the image its too big to display
carro = cv2.resize( carro, ( int(c*0.3),int(r*0.3) ) , interpolation = cv2.INTER_AREA)

#transforming to HSV color spcae
iHSV = cv2.cvtColor(carro, cv2.COLOR_BGR2HSV)

#using the second layer, saturation
iV = iHSV[:,:,2]

#if saturation is less than a threshold then set it to 0
iV[iV<180]=0
iV[iV>0]=255

cv2.imshow('Placa 1',iV)

#this useful transformation lets me create a 3 layered image without using
#numpy reshape.
#i have a layer with shape r,c -> transpose -> c,r.
#stacking it 3 times we have 3,c,r -> transposing this -> r,c,3 which is the form
#that cv2 uses for its images
iV=np.array([iV.T,iV.T,iV.T]).T

#taking the original image and where the mask (the processed layer) is 0 set the
#original image to 0
carro[iV==0]=0

cv2.imshow('Placa 2',carro)

cv2.waitKey(0)
cv2.destroyAllWindows()