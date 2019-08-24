# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 14:43:01 2019

@author: luisf
"""
# =============================================================================
# DIP CLASS 3
#
#P1: Opening and Using Video Processing
# =============================================================================


import cv2
import numpy as np

#this line creates a cature video object
video=cv2.VideoCapture(0)

#while is openned do something
while(video.isOpened()):
    #reads if read was succesful and frames
    ret, frameBGR=video.read()
    if ret!=True:
        break;
    #if succesful does something
    #like showing frames on a window
    
    #BRG to Gray scale
    frameHSV=cv2.cvtColor(frameBGR,cv2.COLOR_BGR2HSV)
    frameGray=cv2.cvtColor(frameBGR,cv2.COLOR_BGR2GRAY)
    
    #concatenating BGR and Gray on a single frame
    frame_color=np.concatenate((frameBGR,frameHSV), axis=1)
    
    
    cv2.imshow('frame1',frame_color)
    cv2.imshow('frame2',frameGray)
    
    #checks a typed key
    tecla = cv2.waitKey(1) & 0xFF
    #if the key is q finishes
    if (tecla == ord('q')):
        #Important to release the video camera and destroy all windows
        video.release()
        cv2.destroyAllWindows()
        break

