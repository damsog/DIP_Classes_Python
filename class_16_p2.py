# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 18:34:59 2019

@author: luisf
"""

# =============================================================================
# DIP with openCV Class 16
#
#P2:    Textures 1
#       Gabor Filters
# =============================================================================

# Gabor filters are a special kind of bandpass filters that are used on 
# numerous problems such as texture analysis, feature extraction and many others
# A Gabor filter gives the highest response at edges and at 
# points where texture changes.

# we can apply this filter doing the convolution with the image like we have
# done with other filters

import cv2
import numpy as np

root = 'D:/U de A/PDI/DIP_Clases_Python/'
img = cv2.imread(root + 'images/river.jpg')
r, c, l =img.shape
img = cv2.resize(img, (int(c*0.5),int(r*0.5)) , cv2.INTER_AREA)

#lets define the kernel of the filter
#sigma is the standard deviation of the Gaussian function used in the Gabor filter.
#theta is the orientation of the normal to the parallel stripes of the Gabor function.
#lambda is the wavelength of the sinusoidal factor in the above equation.
#gamma is the spatial aspect ratio.
#psi is the phase offset.
#ktype indicates the type and range of values that each pixel in the Gabor kernel can hold.

# the syntax to create the filter is:
#gabor = cv2.getGaborKernel(ksize, sigma, theta, lambdaa, gamma, psi, ktype)


# Lets now create various filters with differents parameter to observe
# the differences.

# theta changes the orientation of the filter.lets show 3 different orientations
thetas = [0, np.pi/4, np.pi/2]

# sigma is the std deviations of the filter. if its higher the bandpass of
# the filter will be higher
sigmas = [2.0, 3.0, 4.0]

# a third interesting parameter is lambda which indicates the wavelength of the
# filter window. if its low the filter will be small but more numerous windows
# will be used, andif its higher the window will be bigger
lambd = 6.0

nr, nc, nl = img.shape

#lets show 9 cases changing theta and sigma
for sig in sigmas:
    stack_toshow = np.zeros((1,nc*3,3), np.uint8)
    
    for ang in thetas:
        #creating the filter
        gabor = cv2.getGaborKernel((30,30), sig, ang, lambd, 0.5, 0, ktype = cv2.CV_32F)
        
        #creating an image to show the filter visualy
        gabor_toshow = cv2.resize(gabor, ( nc,nr ), interpolation = cv2.INTER_AREA)
        gabor_toshow = np.around( np.abs(gabor_toshow)*255/ (np.max(gabor_toshow)) ).astype(np.uint8)
        gabor_toshow = cv2.merge([gabor_toshow,gabor_toshow,gabor_toshow])
        
        ##applying the filter
        img_filt = cv2.filter2D(img, cv2.CV_8UC3, gabor)
        
        #stacking the images to show them
        stacked_row = np.column_stack(( gabor_toshow, np.column_stack(( img,img_filt )) ))
        stack_toshow = np.row_stack(( stack_toshow,stacked_row ))    
    
    cv2.imshow('Gabors vs img vs filtered (for sigma ='+str(sig)+')', stack_toshow)
    

cv2.waitKey(0)
cv2.destroyAllWindows()