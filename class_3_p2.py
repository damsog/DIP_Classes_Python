# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:17:49 2019

@author: luisf
"""

# =============================================================================
# DIP CLASS 3
#
#P2: Fusing images
# =============================================================================

import cv2
import os
import numpy as np

root = os.path.abspath(os.getcwd()) + '/images' 

#Before Fusing the images.
#As we know each image (in BGR color system) is form by 3 layers (Blue, Green,Red)
#And each layer is an array with a value from 0 to 255 to represent the grade
#of that color layer. Because the range is 0-255 they are represented as uint8
#unsigned int of 8 bits. IF we cross this limit (overflow the variuable) it
#will ignore the overflow bits. we have to be careful to do numerical operations
#to evade this.

a = cv2.imread(root + 'images/imagen_1.jpg')
b = cv2.imread(root + 'images/imagen_2.jpg')

#casting to uint32 to do the sumation and casting it again o uint8 after the 
#division
c = ( ( a.astype(np.uint32) + b.astype(np.uint32) ) /2).astype(np.uint8)

#Showing the images fused as the average of the two
cv2.imshow('frame1',c)

finished=False
i=0
c=True

#we can change the grade or the fusion so one image can have more
#presence than the other
#d = b*i + a*(1-i)

#Here i made an infinit loop oscilating the grade of the fusion
while(not finished):
    if(c):
        i=i+0.01
        if(i>=1):
            c=False
    else:
        i=i-0.01
        if(i<=0):
            c=True
    #changing the grade i and displaying it on a window
    d = (  b*i  +  a*(1-i)  ).astype(np.uint8)
    cv2.imshow('frame2',d)
   
    #stops the loop with the key q
    tecla = cv2.waitKey(1) & 0xFF
    if(tecla == ord('q')):
        break

#now lets do the negative of an image
na=255-a
nb=255-b
total=np.concatenate( ( np.concatenate( (a,na) , axis=1 ),
                       np.concatenate( (b,nb) , axis=1 )) , axis=1)
cv2.imshow('frame3',total)


#this is different from grayscale. a negative is an image with 3 layers, they
#are just the complement 
#while a grayscale image only has 1 layer and the number is the intensity

#lets do the grayscale to compare
abw=cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
bbw=cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

total_gray=np.concatenate( (abw,bbw) ,axis=1)
cv2.imshow('frame4', total_gray)
    
cv2.waitKey(0)
cv2.destroyAllWindows()