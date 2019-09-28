# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 21:41:14 2019

@author: luisf
"""

# =============================================================================
# DIP with openCV Class 17
#
#P1:    Textures 2: Texture Segmentation Clustering
#         Gabor filter Descriptors (Energy)
# =============================================================================

# Using features extracted from an image using the methods from the last class
# we can use Machine Learning algorithms to clasify images or segment regions on
# an image

# Here I will use the clusterting method to group the pixels on an image
# I intend that each group of pixels are of the same texture to divide an image
# according to its texture.

# each pixel needs a set of descriptors that are unique to each texture
# and gabor filters are useful for that purpose

# First we need to create matrix of features. Nxp, N data, with p features
# each data would be each pixel.

import cv2
import numpy as np
from scipy import stats

# Local Energy
def localEnergy(img, wsize = 11):
    r_img , c_img = img.shape

    win_size = wsize
    
    r_ini = int( (win_size - 1)/2 )
    c_ini = int( (win_size - 1)/2 )
    r_end = r_img - r_ini
    c_end = c_img - c_ini
    
    rows = 0
    cols = 0
    
    # the new image is 10x10 smaller than the original
    img_eng = np.zeros((r_img - r_ini*2 ,c_img - c_ini*2), np.float32)
    
    #passing the window through all the image
    for i in range(r_ini, r_end ):
        cols = 0
        for j in range(c_ini, c_end):
            window = img[ i - r_ini : i + r_ini , j - c_ini : j + c_ini ]
            
            # calculate the mean and std of the window
            mean = cv2.mean(window)[0]
            #img_eng[rows,cols] = np.sum( np.abs( window - mean ) ) / (win_size*win_size)
            img_eng[rows,cols] = np.sum( np.abs( window - mean ) ) / (np.sum(window))

            cols = cols + 1
        
        rows = rows + 1
    
    #normalizing, rounding and casting to uint8 which is the format for images
    img_eng = cv2.copyMakeBorder(img_eng, r_ini, r_ini, c_ini, c_ini, cv2.BORDER_REPLICATE)
    
    return img_eng

# Reading the image
root = 'D:/U de A/PDI/DIP_Clases_Python/images/'
img = cv2.imread(root + '5textures.png')
img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Creating the Gabor filter bank
#gabor = cv2.getGaborKernel(ksize, sigma, theta, lambdaa, gamma, psi, ktype)
ksizes = [10, 50]

sigmas = 0.7
thetas = [np.pi, 3*np.pi/4,np.pi/2,np.pi/4, 0]
#thetas = [np.pi, np.pi/6]
lambd = 8.0

filtered = []
nf = 0
for size in ksizes:
    for th in thetas:
        gabor = cv2.getGaborKernel((size,size), sigmas, th, lambd, 5, 0, ktype = cv2.CV_32F)
        fimage = cv2.filter2D(img_g, cv2.CV_8UC3, gabor)
        # gabor filters alone are not that helpful, so we have to extract 
        #the features from the filteres. here lets use Local energy
        eimage = localEnergy( fimage,25 )
        filtered.append( np.float32(eimage) )
        nf = nf + 1

# nf: number of features
# merges all the images as channels, so we end up with a r,c,nf (nf filters)
features = cv2.merge(filtered)

# reshapes the bank so we end up with a rXc,12. in other words, each new row
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

