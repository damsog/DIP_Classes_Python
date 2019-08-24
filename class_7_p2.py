# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:53:07 2019

@author: luisf
"""


# =============================================================================
# DIP CLASS 7
#
#P2: Separating a character
#    
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

root = 'D:/U de A/PDI/DIP_Clases_Python/'

#loading the image and changing it to gray scale
text = cv2.imread(root + 'images/texto2.bmp')
text = cv2.cvtColor(text, cv2.COLOR_BGR2GRAY)
r,c = text.shape

#inverting bright areas and dark areas (tomake it white text on black background)
#then applying a threshold
text = 255 - text
text[text<100]=0
text[text>0]=255

#increasing the size of the image
textG = cv2.resize(text , (c*4,r*4), interpolation=cv2.INTER_AREA)

#First we want to know where on the image are the text rows for that
#lets get the sum along the columns in the image,leaving only rows with
#the results
sumt = np.sum(textG, axis=1)

#showing the result. the text rows are where the plot has a considerable value
fig, ax = plt.subplots(2,1)
ax[0].imshow(textG)
ax[1].plot(sumt)
#cv2.imshow('Text ',textG)

#now that we know where are the text rows lets cut the image to only the 
#second row
rg,cg =  textG.shape
trow = textG[42:80,:]
#cv2.imshow('Text row 1',trow)

#now we want to know where are the characters on the image
#so lets sum the values along the rows in the image leaving the result for
#column
sumt_row = np.sum(trow, axis=0)

#lets show the plot comparing with the image
fig2, ax2 = plt.subplots(2,1)
ax2[0].imshow(trow)
ax2[1].plot(sumt_row)

#we can see the spikes on vertical letters and between the characters are 
#pits. using this we can separate for example the letter M.
letter = trow[:,16:54]
cv2.imshow('Letter ',letter)


cv2.waitKey(0)
cv2.destroyAllWindows()