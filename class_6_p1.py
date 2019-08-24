# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 19:13:05 2019

@author: luisf
"""
# =============================================================================
# DIP CLASS 6
#
#P1: Morphology: Erosion and Dilatation
# =============================================================================

#Dilatation and erosion are operations that are performed using a kernel or mask
#this kernel (often a square with an anchor point at the center) is compared 
#with the image pixel by pixel and if the anchor point finds something expands
#the image (increases brightness) or reduces it in case of erosion

import cv2
import numpy as np
from time import sleep

root = 'D:/U de A/PDI/DIP_Clases_Python/'

#loading the image and creating a copy to compare
figs = cv2.imread(root + 'images/figuras.tif')
cv2.imshow('Original ',figs)
figs2 = np.copy(figs)

#creating the kernel. thi can be done with a numpy array but cv2 has 
#some build in structures like squares ellipses, crosses etc
#here i will try a cross for testing
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

#loop variables
i = 0
#true ascending, false descending
c = True
finished = False

while(not finished):
    if(c):
        #ascending, dilation
        figs2=cv2.dilate(figs2,kernel, iterations = 1)
        #sleeping for 200ms to apreciate the effect
        sleep(0.2)
        i = i + 1
        if( i>=5 ):
            c=False        
    else:
        #descending, erosion
        figs2=cv2.erode(figs2,kernel, iterations = 1)
        #sleeping for 200ms to apreciate the effect
        sleep(0.2)
        i = i - 1
        if( i<=0 ):
            c=True
    #shows the applied effect
    cv2.imshow('Erosion vs Dilation',figs2)
    
    #stops the loop with the key q
    tecla = cv2.waitKey(1) & 0xFF
    if(tecla == ord('q')):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()