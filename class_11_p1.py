# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:53:25 2019

@author: luisf
"""

# =============================================================================
# DIP with openCV Class 11
#
#P1: top-hat and bot-hat increase constrast
#
# =============================================================================

import cv2
import os
import numpy as np

root = os.path.abspath(os.getcwd()) + '/images' 

nino = cv2.imread('images/nino.bmp')
nino = cv2.cvtColor(nino, cv2.COLOR_BGR2GRAY)
r,c = nino.shape

#nino = cv2.resize(nino,(c*2,r*2),interpolation = cv2.INTER_AREA)

#using a big disk as kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12))

cv2.imshow('Original ',nino)

# getting tophat and blackhat
nino_bh = cv2.morphologyEx(nino, cv2.MORPH_BLACKHAT, kernel)
nino_th = cv2.morphologyEx(nino, cv2.MORPH_TOPHAT, kernel)

cv2.imshow('Blackhat - Tophat', np.column_stack((nino_bh,nino_th)) )

# op(img) = (img + th(img)) - bh(img)

nnino = cv2.add(nino,nino_th)
nnino = cv2.subtract(nnino,nino_bh)

cv2.imshow('New Nino',nnino)

cv2.waitKey(0)
cv2.destroyAllWindows()
