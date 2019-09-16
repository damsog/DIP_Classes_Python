# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 17:16:28 2019

@author: luisf
"""

# =============================================================================
# DIP with openCV Class 12
#
#P1: Transformations 1: Convolution
#               Filtering
# =============================================================================

# -----------------------------------------------------------------------------
# a brief description of a convolution:
# Convolution is one of the most important subjects of systems analysis,
# Signal processing, control etc.
# in summary a convolution is an operation that tells us the response of a
# system to a particular input.    input ----> |Sytem | ---> response.
# lets represent th convolution with X. the reponse of the system h(t) to the 
# input x(t) would be: x(t) X h(t) = y(t)
# a convolution is NOT a multiplication.
# on continous time, a convolution is an integral a bit difficult to
# calculate in some cases, but in discrete time is a sumation.
# the response of the system to the unit impulse is known as the unit impulse
# response h(t) and is what we use to describe a system. d(t) X h(t) = h(t)
# if we transform our space from time to frequency the convolution becomes
# a simple multiplication, and the frequency transform of h(t) its called
# the transfer function of the system H(w) (w is ohmega)
# X(W) X H(W) = Y(W)
#------------------------------------------------------------------------------
# Now back to DIP. 
# if the system is our image and our input is a filter, if we convolve
# the image and a filter we get the response of the system which is the
# image filtered (the convolution is commutable, doesn't matter the order)
# convolution on images works similar to signals but on 2D intead of only 1

# As in one-dimensional signals, images also can be filtered with various 
# low-pass filters(LPF), high-pass filters(HPF) etc.

import cv2
import numpy as np

# lets make a simple averaging filter
# for that lets use a normalized box filter
kernel = np.ones((10,10), np.float32)/100

root = 'D:/U de A/PDI/DIP_Clases_Python/'
img = cv2.imread(root + 'images/placa_p.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# lets convolve the image and the kernel of the filter with anchor point at
# the center
img_avg = cv2.filter2D(img, -1, kernel)

cv2.imshow('AVG: ORG vs Filtered', np.column_stack((img, img_avg)))

# letsuse gausian blurr. this filter blurres the image using a gaussian function
img_blur = cv2.GaussianBlur(img, (5,5), 0)
cv2.imshow('Gauss: ORG vs Filtered', np.column_stack((img, img_blur)))

# now lets use median blurring. this one takes the median of all the pixels and
# sets the kernel of the filter as the median.
# really useful when dealing with salt-and-pepper kind of noise
# lets create some noise for our image
noise = (img.copy()*0).astype(np.int8)
cv2.randn(noise,-100,255)

noise = noise + img
noise[noise<0] = 0
noise = noise.astype(np.uint8)

# now lets filter the noise
img_med = cv2.medianBlur(noise, 5)
cv2.imshow('Med: Noisy vs Filtered', np.column_stack((noise, img_med)))

# for last lets take a look a bilateral filter, which filters the noise
# on an image but keeping edges intact. but is also slower to other filters
img_bil = cv2.bilateralFilter(noise, 9,75,75)
cv2.imshow('Bilateral: Noisy vs Filtered', np.column_stack((noise, img_bil)))


cv2.waitKey(0)
cv2.destroyAllWindows()
