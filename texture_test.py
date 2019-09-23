# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 21:41:14 2019

@author: luisf
"""

# =============================================================================
# DIP with openCV Testing reducing levels codification
#
# =============================================================================

import cv2
import numpy as np
import os

root = 'D:/U de A/PDI/DIP_Clases_Python/images/textures_tests/'
levels = 4

for name in os.listdir(root):
    img  =  cv2.imread(root + name)
    div = int( 256 / levels )
    img_b =  np.floor(img/div).astype(np.uint8)
    img_b =  ((img_b / np.max(img_b))*255).astype(np.uint8)
    
    cv2.imshow('Image: '+ name ,img_b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

