# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 19:50:12 2019

@author: luisf
"""


# =============================================================================
# DIP CLASS 4
#
#P1:   Movement detection using area
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

#creating a video object
video = cv2.VideoCapture(0)
area_t=[]

#capturing 50 frames
for i in range(50):
    
    #for this we capture a video image on an instant and an instant later
    ret, frame1=video.read()
    ret2, frame2=video.read()
    if(ret!=True):
        break
    
    #then we subtract. the frames
    #with this first form the subtraction doesn't saturate, if is less than
    #0 it wraps around (underflows) and becomes something big due to the negative
    #on unsigned integers
    
    #frame3 = frame1 - frame2
    
    #the cv2 method has saturation, if we surpass 0 or 255 it saturates to those values
    frame3 = cv2.subtract(frame1,frame2)
    
    #we sum the area of the whole frame and stack it
    area = np.sum(frame3)
    area_t.append(area)
    
    cv2.imshow('Frame',frame3)
    
    #to finish early with the key q
    fin = cv2.waitKey(1) & 0xFF
    if(fin == ord('q')):
        break

#plotting the area changes on time. the spikes indicate momevent
plt.plot(area_t)
video.release()
cv2.destroyAllWindows()