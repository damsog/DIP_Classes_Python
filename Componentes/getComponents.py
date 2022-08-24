# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 15:34:50 2019

@author: luisf
"""

# =============================================================================
#       Conversion to other systems function
# =============================================================================
import cv2
import os
import numpy as np

def components( imagen):
    r,c,l = imagen.shape
    if(l < 3):
        return imagen
    
    iBGR = imagen
    iHSV = cv2.cvtColor(iBGR, cv2.COLOR_BGR2HSV)
    iLab = cv2.cvtColor(iBGR, cv2.COLOR_BGR2Lab)
    iLuv = cv2.cvtColor(iBGR, cv2.COLOR_BGR2Luv)
    iHLS = cv2.cvtColor(iBGR, cv2.COLOR_BGR2HLS)
    #iHSV = cv2.subtract(iHSV,255)
    iBGR = ( 255 - iBGR )
    #B~ , S, b, v, L
    return iBGR[:,:,0], iHSV[:,:,1], iLab[:,:,2], iLuv[:,:,2], iHLS[:,:,1]

    