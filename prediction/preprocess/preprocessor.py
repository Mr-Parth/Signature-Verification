import cv2 as cv
import os
import numpy as np
import warnings
warnings.simplefilter("ignore")

def normalize(img):
    img = img/255
    return img

def filter_n_scale(image):

    image=cv.resize(image,(258,153),interpolation= cv.INTER_AREA)
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
     
    inv=np.invert(gray)
    
    return inv




