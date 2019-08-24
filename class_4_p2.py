# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 20:45:08 2019

@author: luisf
"""

# =============================================================================
# DIP CLASS 4
#
#P1:   Different Color Systems
# =============================================================================

# There are different systems to process color digitally. RGB for example
# uses a coordenate system on a space where each coordenate (R, G, B) has a 
# value and the color would be a point in said space. HSV on the other hand
# is like a cilinder like space, where value is like Z, saturation ir r, and
# hue, which is the color itself, is the angle.
# there are numerous other systems for representing color like CMYK, or Lab

import cv2
import numpy as np

root = 'D:/U de A/PDI/DIP_Clases_Python/'

#reading an image and transforming from BGR to other systems
pBGR = cv2.imread(root + 'images/peppers.jpg')
pHSV = cv2.cvtColor(pBGR, cv2.COLOR_BGR2HSV)
pLab = cv2.cvtColor(pBGR, cv2.COLOR_BGR2Lab)

#comparing BGR to HSV
cv2.imshow('Peppers BGR vs HSV',np.column_stack((pBGR,pHSV)))

s=pHSV[:,:,1]

#re-scalling the image to fit the 9 layers to compare
r,c,l=pHSV.shape
pHSVs = cv2.resize(pHSV, (int(c*0.5),int(r*0.5) ) ,interpolation = cv2.INTER_AREA)
pBGRs = cv2.resize(pBGR, (int(c*0.5),int(r*0.5) ) ,interpolation = cv2.INTER_AREA)
pLabs = cv2.resize(pLab, (int(c*0.5),int(r*0.5) ) ,interpolation = cv2.INTER_AREA)

#separating and displaying each layer of the three systems to compare
compare1 = np.column_stack(( pHSVs[:,:,0], np.column_stack(( pHSVs[:,:,1],pHSVs[:,:,2] )) ))
compare2 = np.column_stack(( pBGRs[:,:,0], np.column_stack(( pBGRs[:,:,1],pBGRs[:,:,2] )) ))
compare3 = np.column_stack(( pLabs[:,:,0], np.column_stack(( pLabs[:,:,1],pLabs[:,:,2] )) ))
compare = np.row_stack(( compare1, np.row_stack(( compare2,compare3 )) ))
cv2.imshow('Layers H-S-V | B-G-R | L-a-b', compare)



cv2.waitKey(0)
cv2.destroyAllWindows()