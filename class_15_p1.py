# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 11:42:54 2019

@author: luisf
"""

# =============================================================================
# DIP with openCV Class 15
#
#P1: Transformations 3: Spectral Transformations
#     Discrete Fourier and Cosine Transforms for 2D arrays
# =============================================================================

# The fourier transform its an operation that we can apply on a non periodic
# function that allows ud to change from the time space to the frequency space.
# the discrete fourier transform its the operation that we have to apply to
# obtain a discrete spectrum.

import cv2
import os
import numpy as np

root = os.path.abspath(os.getcwd()) + '/images' 
img = cv2.imread(root + 'images/imagen_2.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
r,c = img.shape

# The fast fourier transform is an algorithm widely used to calculate the DFT
# efficiently and it works better for certain sizes. similar to 1D signals 
# that its more efficient for 2^n sizes the algorithm used on opencv works 
# better for certain sizes of images.

#optimal number of rows and cols
opr = cv2.getOptimalDFTSize(r)
opc = cv2.getOptimalDFTSize(c)

#img padding to optimal size. img, t, b, l, r, borderType
img_padd = cv2.copyMakeBorder(img, 0, opr - r, 0, opc - c, cv2.BORDER_CONSTANT)

# creates an array of the size of the image of zeros and stacks them
# in parallel (like layers) using merge. split is the oposite.
# we need 2 layers to get the real and the imaginary part of the DFT
planes = [np.float32(img_padd), np.zeros(img_padd.shape, np.float32) ]
img_complex = cv2.merge(planes)

# gets the DFT. the result is a real part [0] and a complex part [1]
img_dft = cv2.dft(img_complex)

# splits the pplanes again
planes = cv2.split(img_dft)

# obtains the magnitud of the complex matrix
dft_mag = cv2.magnitude(planes[0], planes[1])

# the values are too big so we have to scale them

dft_mag = 20*cv2.log(dft_mag)

# Cropping the img for some reason
dft_mag = dft_mag[0:(dft_mag.shape[0] & -2), 0:(dft_mag.shape[1] & -2)]
hr = int(dft_mag.shape[0]/2 )
hc = int(dft_mag.shape[1]/2 )

#----- Swapping spaces-----
#its the same as to use: np.fft.fftshift(dft_mag)
q0 = dft_mag[0:hr,0:hc]   #
q1 = dft_mag[0:hr,hc:hc*2]
q2 = dft_mag[hr:hr*2,0:hc]
q3 = dft_mag[hr:hr*2,hc:hc*2]

temp1 = q0.copy()
dft_mag[0:hr,0:hc] = q3
dft_mag[hr:hr*2,hc:hc*2] = temp1

temp1 =  q1.copy()
dft_mag[0:hr,hc:hc*2] = q2
dft_mag[hr:hr*2,0:hc] = temp1
#----------------------------

dft_mag = cv2.normalize(dft_mag, 0, 255, cv2.NORM_MINMAX)


cv2.imshow('ORG', img)
cv2.imshow('DFT', dft_mag)

#------------------------------------------------------------------------------
# The DFT operates with complex numbers, so each value has a real part and
# an imaginary part. and for us to work with sometimes we have to compute the 
# magnitud of the 2. the DCT is only the real part of the DFT, which is useful
# to work with when we are only interested with real values.

dct = cv2.dct(np.float32(img))
dct = 20*cv2.log( dct )

dct = np.fft.fftshift(dct)

cv2.imshow('DCT', dct)

cv2.waitKey(0)
cv2.destroyAllWindows()