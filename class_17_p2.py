# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 19:14:49 2019

@author: luisf
"""

# =============================================================================
# DIP with openCV Class 17
#
#P1:    Textures 2: Texture Segmentation Clustering
#                Gabor Filter Descriptors (Magnitud)
# =============================================================================

import cv2
import numpy as np
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

# Reading the image
root = 'D:/U de A/PDI/DIP_Clases_Python/images/'
img = cv2.imread(root + '5textures.png')
img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Creating the Gabor filter bank
#gabor = gabor_kernel(frequency, theta=0, bandwidth=1, sigma_x=None, sigma_y=None, n_stds=3, offset=0)

sigmas = [9]
thetas = [np.pi, np.pi/2, 0]
#thetas = [np.pi, np.pi/6]
freq = [0.25, 0.3]

filtered = []
nf = 0
for th in thetas:
    for sigma in sigmas:
        for frequency in freq:
            gabor = np.real(gabor_kernel(frequency, theta=th,
                                         sigma_x=sigma, sigma_y=sigma))
            
            
            # gabor filters alone are not that helpful, so we have to extract 
            #the features from the filteres. here lets use the magnitude
            mimage = np.sqrt(ndi.convolve(img_g, np.real(gabor), mode='wrap')**2 + 
                             ndi.convolve(img_g, np.imag(gabor), mode='wrap')**2)
            filtered.append( np.float32(mimage) )
            nf = nf + 1

# nf: number of features
# merges all the images as channels, so we end up with a r,c,nf (nf filters)
features = cv2.merge(filtered)

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

