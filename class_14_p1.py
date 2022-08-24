# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 16:48:41 2019

@author: luisf
"""

# =============================================================================
# DIP with openCV Class 14
#
#P1: Using What we have learned to separate a part of an image
#
# =============================================================================

import cv2
import os
import numpy as np

def layers2single(img):
    #this function takes an image with 3 layers (color) and
    #stacks them on a single serial layer
    shape = img.shape
    if( len(shape) < 3 ):
        return img
    r,c,l = img.shape
    img = np.column_stack((img[:,:,0], np.column_stack(( img[:,:,1],img[:,:,2] )) ))    
    
    return img
def single2layers(img):
    #this image does the oposite of the above function
    #takes a single layered image cuts it on 3 sections
    #and stacks them in parallel (back to 3 layers)
    shape = img.shape
    if( len(shape) > 2 ):
        return img
    r,c = img.shape
    nc = int(c/3)
    imgl1 = img[: , 0:nc ]
    imgl2 = img[: , nc:nc*2 ]
    imgl3 = img[: , nc*2:nc*3 ]
    #i have 3 images r,c and i want to stack them to get an image
    #that should be r,c,l. so i transpose each one to c,r and stack them
    #getting 3,c,r then i transpose the image obtaining r,c,3 
    img = np.array([imgl1.T,imgl2.T,imgl3.T]).T
    return img

def avgfilter(img):
    kernel = np.ones((10,10), np.float32)/100
    anchor = (0,0)
    nimg = cv2.filter2D(img,-1, kernel, anchor)
    return nimg

def decompose(img, printcomps = False):
    r,c,l = img.shape
    iBGR = img
    iHSV = cv2.cvtColor(iBGR, cv2.COLOR_BGR2HSV)
    iLab = cv2.cvtColor(iBGR, cv2.COLOR_BGR2Lab)
    iLuv = cv2.cvtColor(iBGR, cv2.COLOR_BGR2Luv)
    iHLS = cv2.cvtColor(iBGR, cv2.COLOR_BGR2HLS)
    #iHSV = cv2.subtract(iHSV,255)
    iBGR = ( 255 - iBGR )
    iHSV = layers2single(iHSV)
    iLab = layers2single(iLab)
    iLuv = layers2single(iLuv)
    iHLS = layers2single(iHLS)
    iBGR = layers2single(iBGR)
    if(printcomps == True):
        iHSVs = cv2.resize(iHSV, (int(c*0.15),int(r*0.15)), interpolation= cv2.INTER_AREA )
        iLabs = cv2.resize(iLab, (int(c*0.15),int(r*0.15)), interpolation= cv2.INTER_AREA )
        iLuvs = cv2.resize(iLuv, (int(c*0.15),int(r*0.15)), interpolation= cv2.INTER_AREA )
        iHLSs = cv2.resize(iHLS, (int(c*0.15),int(r*0.15)), interpolation= cv2.INTER_AREA )
        iBGRs = cv2.resize(iBGR, (int(c*0.15),int(r*0.15)), interpolation= cv2.INTER_AREA )
        cv2.imshow('HSV,Lab,Luv,HLS,BGR', np.row_stack(( np.row_stack (( np.row_stack((iHSVs,iLabs)),
                                                                        np.row_stack((iLuvs,iHLSs)) )), iBGRs ))  )
    return iHSV, iLab, iLuv, iHLS, iBGR


root = os.path.abspath(os.getcwd()) + '/images' 

img = cv2.imread(root + '/Class12/015.jpg')
r,c,l = img.shape

img_avg = avgfilter(img)

#HSV,Lab,Luv,HLS,BGR = decompose(img_avg,printcomps=True)
img_hsv = cv2.cvtColor(img_avg, cv2.COLOR_BGR2HSV)
S = img_hsv[:,:,1] #S layer

_,S_thresh = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow('Thresholded',S_thresh)

#filling the holes on the image
maskfill = np.zeros((r+2,c+2),np.uint8)
S_fill = S_thresh.copy()
# Floodfill basicly fills (sets to the value we input) the pixels it finds
#to be zero (but wont set to max if these values are inside a nonzero figure)

#lets make a mask setting all the background to max
cv2.floodFill(S_fill, maskfill, (0,0), 255)
#only the holes inside the figure are not filled
#so now lets invert the image so the holes are now the
#only thing set to 1
S_fill = cv2.bitwise_not(S_fill)
S_thresh = S_thresh | S_fill

#cv2.imshow('Holes filled ',S_thresh)

contours,_ = cv2.findContours(S_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#lets find the biggest figure on the layer and set 0 everything else
S_n = img.copy()*0
area_max = 0
i_max = 0
for i in range(len(contours)):
    cv2.drawContours(S_n, contours, i , (255,255,255), -1)
    area = np.sum(S_n) /255
    if(area > area_max):
        area_max = area
        i_max = i
    S_n = S_n*0

S_thresh = S_thresh*0
#now that we have the biggest one lets draw it and fill it
cv2.drawContours(S_thresh, contours, i_max, (255,255,255), -1)

#stack the layer 3 times horizontally to pass it to our functtion that cuts
#it on 3 and stacks the layers parallel 
#(The 3 layers are the same but we need 3 to compare it to the color image)
S_thresh = np.column_stack(( S_thresh , np.column_stack(( S_thresh,S_thresh )) ))
S_thresh = single2layers(S_thresh)

#we set to 0 where our layer mask is 0
img[S_thresh==0] = 0
cv2.imshow('Img masked',img)
cv2.imshow('Mask ', S_thresh)

cv2.imwrite(root+'images/class12/imageMasked.jpg',img)
cv2.imwrite(root+'images/class12/imageMask.jpg',S_thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()