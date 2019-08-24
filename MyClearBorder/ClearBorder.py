# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 15:11:34 2019

@author: luisf
"""

import numpy as np

def myclearborder(image, porcentager, porcentagec):
    r, c =image.shape
    if((porcentager>=1) or (porcentager<=0) or (porcentagec>=1) or (porcentagec<=0)):
        return image
    rw = int(r*porcentager)
    cw = int(c*porcentagec)
    image[:,0:cw] = 0
    image[:,(c-cw):cw] = 0
    image[0:rw,:] = 0
    image[(r-rw):r,:] = 0
    return image