# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:54:07 2019

@author: luisf
"""


# =============================================================================
# DIP with openCV Class 15
#
#P2: Transformations 3: Spectral Transformations
#   Convolution and Multiplications on the spectral domain
# =============================================================================

# As we saw on a previous chapter, the convolution operation x(t) X h(t) = y(t)
# on the frequency domain becomes a simple multiplication X(w)H(w) = Y(w)
# which makes the calculation of a convolution far more easy not only manually
# but also computationally

import cv2
import os
import numpy as np

root = os.path.abspath(os.getcwd()) + '/images' 
img = cv2.imread(root + 'images/placa_p.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
r,c = img.shape
rc = int(r/2)
cc = int(c/2)

# LPF
fw = 40

kernel = np.zeros((img.shape[0],img.shape[1],2), np.float32)
kernel[rc-fw:rc+fw,cc-fw:cc+fw] = 1

#creating 2 layer fror the real and imaginary part and calculating the dft
planes = [np.float32(img) , np.zeros(img.shape,np.float32)]
planes = cv2.merge(planes)
dft_img = cv2.dft(planes)
#shifting to correct the low f
dft_img = np.fft.fftshift(dft_img)


#res = cv2.mulSpectrums(dft_img,k_dft, 0)
#applying the filter
res = dft_img*kernel
res = np.fft.fftshift(res)

#inverse DFT and normalizing
idft_img = cv2.idft(res)
img_fil = cv2.magnitude( idft_img[:,:,0],idft_img[:,:,1] )
img_fil =( img_fil /(np.max(img_fil)) *255 ).astype(np.uint8)


cv2.imshow('ORG vs Filtered', np.column_stack((img, img_fil)))


cv2.waitKey(0)
cv2.destroyAllWindows()