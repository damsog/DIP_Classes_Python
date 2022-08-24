# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 12:05:09 2019

@author: luisf
"""

# =============================================================================
# DIP with openCV Class 17
#
#P1:    Textures 2: Texture Segmentation Clustering
#                GLCM Descriptors
# =============================================================================

import cv2
import os
import numpy as np
from skimage import feature

# GLCM transforms
def GLCMtransforms(img, wsize = 11):
    r_img , c_img = img.shape

    win_size = wsize
    
    r_ini = int( (win_size - 1)/2 )
    c_ini = int( (win_size - 1)/2 )
    r_end = r_img - r_ini
    c_end = c_img - c_ini
    
    rows = 0
    cols = 0
    
    # the new image is 10x10 smaller than the original
    img_cont = np.zeros((r_img - r_ini*2 ,c_img - c_ini*2), np.float32)
    img_diss = np.zeros((r_img - r_ini*2 ,c_img - c_ini*2), np.float32)
    img_hom = np.zeros((r_img - r_ini*2 ,c_img - c_ini*2), np.float32)
    img_eng = np.zeros((r_img - r_ini*2 ,c_img - c_ini*2), np.float32)
    img_corr = np.zeros((r_img - r_ini*2 ,c_img - c_ini*2), np.float32)
    img_asm = np.zeros((r_img - r_ini*2 ,c_img - c_ini*2), np.float32)

    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    #passing the window through all the image
    for i in range(r_ini, r_end ):
        cols = 0
        for j in range(c_ini, c_end):
            window = img[ i - r_ini : i + r_ini , j - c_ini : j + c_ini ]
            
            # Calculate the GLCM for the window
            w_GLCM = feature.greycomatrix(window, [1], [angles[0]],levels=levels)
            img_cont[rows, cols] = feature.greycoprops(w_GLCM, 'contrast')
            img_diss[rows, cols] = feature.greycoprops(w_GLCM, 'dissimilarity')
            img_hom[rows, cols] = feature.greycoprops(w_GLCM, 'homogeneity')
            img_eng[rows, cols] = feature.greycoprops(w_GLCM, 'energy')
            img_corr[rows, cols] = feature.greycoprops(w_GLCM, 'correlation')
            
            cols = cols + 1
        
        rows = rows + 1
    
    #normalizing, rounding and casting to uint8 which is the format for images
    img_cont = cv2.copyMakeBorder(img_cont, r_ini, r_ini, c_ini, c_ini, cv2.BORDER_REPLICATE)
    img_diss = cv2.copyMakeBorder(img_diss, r_ini, r_ini, c_ini, c_ini, cv2.BORDER_REPLICATE)
    img_hom = cv2.copyMakeBorder(img_hom, r_ini, r_ini, c_ini, c_ini, cv2.BORDER_REPLICATE)
    img_eng = cv2.copyMakeBorder(img_eng, r_ini, r_ini, c_ini, c_ini, cv2.BORDER_REPLICATE)
    img_corr = cv2.copyMakeBorder(img_corr, r_ini, r_ini, c_ini, c_ini, cv2.BORDER_REPLICATE)
    img_asm = img_eng**2
    
    return img_cont,img_diss,img_hom,img_eng,img_corr, img_asm

# Reading the image
root = os.path.abspath(os.getcwd()) + '/images' 
img = cv2.imread(root + '/5textures.png')
img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# the GLCM matrix takes all the posible combinations of changes. for 8 bit
# we have 256x256 which can take some time. to fix this we can reduce our image
# to a lower number of levels to compute faster. 
levels = 8

div = int( 256 / levels )
img_5b =  np.floor(img_g/div).astype(np.uint8)

# Window  size  has  a significant effect on the segmentation accuracy. 
# Large window  size  leads  to  more  stable  texture  features  but tends
# to blur the edges, while small window size leads to   misclassify   the   
# textured   boundaries.   Thus,   an efficient windowing   
# should   be   achieved.
img_cont,img_diss,img_hom,img_eng,img_corr,_ = GLCMtransforms( img_5b, 21)

channels = [img_cont,img_diss,img_hom,img_eng,img_corr]
nf = len(channels)

# merges all the images as channels, so we end up with a r,c,nf (nf channels)
features = cv2.merge(channels)

# reshapes the bank so we end up with a rXc,nf. in other words, each new row
# its a pixel, and each column its a feature, and we have nf features for each
# pixel
features = features.reshape((-1,nf))

#%%
#==============================================================================
#                                   CLUSTERING
# stop criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

clusters = 5
attempts=10

# applying k-means clustering. we get the labels for each cluster for th data
# and the centers of the data
ret,label,center=cv2.kmeans(features,clusters,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

#center =np.uint8(center)

mul = int(255 / (clusters-1) )
img_seg = label.reshape((img_g.shape))
img_seg = np.uint8(img_seg * mul)

cv2.imshow('ORG vs Segmented', np.column_stack(( img_g,img_seg )) )


cv2.waitKey(0)
cv2.destroyAllWindows()