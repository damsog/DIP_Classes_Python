# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:41:39 2019

@author: luisf
"""

# =============================================================================
# DIP CLASS 5
#
#P1: Using Color Layers to separate the license plate on an image
#    using multiple layers from different systems
# =============================================================================

import cv2
import os
import numpy as np
from Componentes.getComponents import components

root = os.path.abspath(os.getcwd()) + '/images' 

carro = cv2.imread(root + '/carro.jpg')

r,c,l = carro.shape

#the image is too big to display so we will scale it a bit
#comment this to process the image at full scale to save it
#carro = cv2.resize( carro, ( int(c*0.1),int(r*0.1) ) , interpolation = cv2.INTER_AREA)

#obtaining layers of different systems (personal funtion)
a, b, c, d, f = components(carro)

#just displaying said layers on a single image
imge1 = np.column_stack((a,b))
imge2 = np.column_stack((c,d))
imge3 = np.column_stack((f,f))
imge1 = np.row_stack((imge1, np.row_stack(( imge2,imge3 )) ))
cv2.imshow('BGR(B Negative) HSV(S) | Lab(b) Luv(v) | HSL(L) HSL(L) ',imge1)

#finding the minimum and maximum components of each layer
min1 = np.minimum(b,d)
max1 = np.maximum(b,d)
min2 = np.minimum(c,d)
max2 = np.maximum(c,d)
min3 = np.minimum(min1,min2)
max3 = np.maximum(max1,max2)

#displaying the resulted minimums and maximums images
imge1 = np.column_stack((min1,max1))
imge2 = np.column_stack((min2,max3))
imge3 = np.column_stack((min3,max3))
imge1 = np.row_stack((imge1, np.row_stack(( imge2,imge3 )) ))
cv2.imshow('Minimums Vs Maximums',imge1)

#taking the resulting layer and seting a threshold to set to 0 useless stuff
min3[min3<160]=0
min3[min3>160]=255
cv2.imshow('Minimum 1',min3)

#comment this if not at full scale to not re-qrite the processed image at full scale
cv2.imwrite(root + '/placaCarro_2.tif',min3)

cv2.waitKey(0)
cv2.destroyAllWindows()

