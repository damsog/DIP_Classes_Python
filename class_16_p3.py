# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 21:12:29 2019

@author: luisf
"""

# =============================================================================
# DIP with openCV Class 16
#
#P3:    Textures 1
#       Gray Level Co-occurrence Matrix (GLCM)
# =============================================================================

# GLCM is a method for caracterizing textures on an image based on the 
# differences of one pixel to a neighbor. Usually we take only 2 neighbor
# pixels because using more would be extremly computanionally demanding.

# a GLCM is a squared matrix which contains the total combinations of changes
# of a pixel value to a different neighbor value fr all the posible combinations

# once we have a GLCM we can summarize it using different methods divided
# mainly in 3 groups

# Contrast -----| Contrast
#               | Dissimilarity
#               | homogeneity

# Orderliness --| Angular Second mooment and Energy
#               | entropy

# Statistical --| Mean
#               | Std
#               | Correlation

# we have to select one of these techniques calculate from our GLCM of a window
# of our image to obtain one value, and sliding the window through the image
# we then create a new texture image.

import cv2
import os
import numpy as np
# opencv doesn't have a GLCM implementation anymore for some reason
# so lets import it from skimage (from sci-kit. another great image processing
# library)

from skimage import feature


root = os.path.abspath(os.getcwd()) + '/images' 
img = cv2.imread(root + '/field.jpg')
img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# the GLCM matrix takes all the posible combinations of changes. for 8 bit
# we have 256x256 which can take some time. to fix this we can reduce our image
# to a lower number of levels to compute faster. 
levels = 8

div = int( 256 / levels )
img_5b =  np.floor(img_g/div).astype(np.uint8)

# image size
r_img,c_img = img_g.shape

# window size
win_size = 13

r_ini = int( (win_size - 1)/2 )
c_ini = int( (win_size - 1)/2 )
r_end = r_img - r_ini
c_end = c_img - c_ini

rows = 0
cols = 0

angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

img_cont = np.zeros((r_img - r_ini*2,c_img - c_ini*2), np.float32)
img_diss = img_cont.copy()
img_hom = img_cont.copy()
img_eng = img_cont.copy()
img_corr = img_cont.copy()


#sliding the window through all the image
for i in range(r_ini, r_end ):
    cols = 0
    for j in range(c_ini, c_end):
        window = img_5b[ i-r_ini : i+r_ini , j-c_ini : j+c_ini ]
        
        # Calculate the GLCM for the window
        w_GLCM = feature.greycomatrix(window, [1], [angles[0]],levels=levels)
        img_cont[rows, cols] = feature.greycoprops(w_GLCM, 'contrast')
        img_diss[rows, cols] = feature.greycoprops(w_GLCM, 'dissimilarity')
        img_hom[rows, cols] = feature.greycoprops(w_GLCM, 'homogeneity')
        img_eng[rows, cols] = feature.greycoprops(w_GLCM, 'energy')
        img_corr[rows, cols] = feature.greycoprops(w_GLCM, 'correlation')
        
        cols = cols + 1
    
    rows = rows + 1

#normalizing, rounding and casting to uint8 which is the format for images.
# only for display
img_cont = ( (img_cont / (np.max(img_cont) ) )*255 ).astype(np.uint8)
img_diss = ( (img_diss / (np.max(img_diss) ) )*255 ).astype(np.uint8)
img_hom = ( (img_hom / (np.max(img_hom) ) )*255 ).astype(np.uint8)
img_eng = ( (img_eng / (np.max(img_eng) ) )*255 ).astype(np.uint8)
img_corr = ( (img_corr / (np.max(img_corr) ) )*255 ).astype(np.uint8)



# the resulting image is smaller than the original duw to the window. so we
# have to create a border that replicates close pixel values

img_cont = cv2.copyMakeBorder(img_cont, r_ini, r_ini, c_ini, c_ini, cv2.BORDER_REPLICATE)
img_diss = cv2.copyMakeBorder(img_diss, r_ini, r_ini, c_ini, c_ini, cv2.BORDER_REPLICATE)
img_hom = cv2.copyMakeBorder(img_hom, r_ini, r_ini, c_ini, c_ini, cv2.BORDER_REPLICATE)
img_eng = cv2.copyMakeBorder(img_eng, r_ini, r_ini, c_ini, c_ini, cv2.BORDER_REPLICATE)
img_corr = cv2.copyMakeBorder(img_corr, r_ini, r_ini, c_ini, c_ini, cv2.BORDER_REPLICATE)


#%%


cv2.imshow('Comparing', np.row_stack(( np.column_stack(( img_g,img_cont )),
                             np.row_stack(( np.column_stack(( img_diss,img_hom )),
                                           np.column_stack(( img_eng,img_corr )) )) )) )

cv2.waitKey(0)
cv2.destroyAllWindows()



