# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:28:38 2019

@author: luisf
"""

# =============================================================================
# DIP with openCV Class 15
#
#P3: Transformations 4: More Transformations 
#                      Integral Images
# =============================================================================

#todo


import cv2
import os
import numpy as np

root = os.path.abspath(os.getcwd()) + '/images' 
img = cv2.imread(root + '/placa_p.jpg')


cv2.waitKey(0)
cv2.destroyAllWindows()