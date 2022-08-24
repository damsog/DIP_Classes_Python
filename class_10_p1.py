# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:31:19 2019

@author: luisf
"""

# =============================================================================
# DIP with openCV Class 10
#
#P1: Intro to contours
# =============================================================================

# Contours is a really important part of CV and openCV. the way that contours
# work on opencv is creating sequences that make up a contour and they can be
# nested on hierachies but here lets just make a simple example of finding
# contours of binary figures

import cv2
import os
import numpy as np

root = os.path.abspath(os.getcwd()) + '/images' 

# lets read the image and keep it as (BGR even tho its binary) to draw the
# contours later using colors
figures = cv2.imread(root + 'images/figuras.bmp')
r,c,l = figures.shape
cfigures = np.copy(figures)
# converting to grayscale and thresholding. findcontours works on binary images
if(l>1):
    figuresg = cv2.cvtColor(figures, cv2.COLOR_BGR2GRAY)
figuresg[figuresg>0]=255

# findcontours receives the image, thestructure to organize the contours and 
# an aproximation option for the points. It returns the image that we passed to
# it, the contours itself and the hierarchy of the contours. lets ignore
# the first one and the last one for now
_,contours,_ = cv2.findContours(figuresg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# function to draw the contours. an image to draw upon, the contour to draw,
# the color, and the thickness of the contour. this modifies the image it draws on!
# if the thickness is -1 it fills the contours
cv2.drawContours(cfigures,contours,-1,(0,255,0),1)

cv2.imshow('Figures and contours ',cfigures)
cv2.waitKey(0)

for i in range(3):
    #creating a mask where to draw the images
    mask = np.zeros(figures.shape[:2] , np.uint8)
    #draws each contour on a new mask filled with white
    cv2.drawContours(mask,contours,i, (255,255,255), -1)
    #calculating the area of a figure each iteration and showing it
    area = np.sum(mask)/255
    cv2.imshow('Figure: '+str(i) +' Area: '+str(area),mask)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()