# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:45:46 2019

@author: luisf
"""


# =============================================================================
# DIP CLASS 7
#
#P3: #analyzing the number plate
#    
# =============================================================================

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
root = os.path.abspath(os.getcwd()) + '/images' 

#loading the number plate and converting to gray scale
plate = cv2.imread(root + '/placa_p.jpg')
plateg = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

#applying threshold
plateg[plateg>115] = 255
plateg[plateg<255] = 0

#inverting so the text is bright and the background black
plateg = 255 - plateg
#summing the values of the rows for each columnleaving only rows with the sums
sump = np.sum(plateg, axis=1)

#we can observe where the text lies. the last spike is the bottom border 
#because is acontinous line along the columns
fig,  ax = plt.subplots(2,1)
ax[0].imshow(plateg)
ax[1].plot(sump)

