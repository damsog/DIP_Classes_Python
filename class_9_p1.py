# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 21:39:18 2019

@author: luisf
"""

# =============================================================================
# DIP CLASS 9
#
#P1: Morphology on GrayScale
#    
# =============================================================================

# We have previously applied morphology operations (dilation, erosion, openning,
# closing,etc) on binary images (0 or 255)
# but we can apply them too to grayscale images
#

import cv2
import os
#import numpy as np
from time import sleep

root = os.path.abspath(os.getcwd()) + '/images' 

carro = cv2.imread('images/carro(1).jpg')
carro = cv2.cvtColor(carro, cv2.COLOR_BGR2GRAY)
r,c = carro.shape
carro = cv2.resize(carro , (int(c*0.3),int(r*0.3)) )

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
cv2.imshow('Carro',carro)

#loop variables
i = 0
#true ascending, false descending
c = True
finished = False

while(not finished):
    if(c):
        #ascending, dilation
        carro = cv2.dilate(carro,kernel, iterations = 1)
        #sleeping for 200ms to apreciate the effect
        sleep(0.2)
        i = i + 1
        if( i>=5 ):
            c=False        
    else:
        #descending, erosion
        carro = cv2.erode(carro,kernel, iterations = 1)
        #sleeping for 200ms to apreciate the effect
        sleep(0.2)
        i = i - 1
        if( i<=0 ):
            c=True
    #shows the applied effect
    cv2.imshow('Erosion vs Dilation',carro)
    
    #stops the loop with the key q
    tecla = cv2.waitKey(1) & 0xFF
    if(tecla == ord('q')):
        break
    
cv2.waitKey(0)
cv2.destroyAllWindows()
