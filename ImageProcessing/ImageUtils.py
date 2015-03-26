# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import math
import cv2
 
 
def PrintImg(im_r, meassage):
   
    plt.figure()
    plt.title(meassage)
    plt.imshow(im_r, cmap = 'gray')
    plt.show()

def ImageRescale(im_r, TARGET_MPIXELS=1e6):   
    height, width, depth = im_r.shape
    mpixels=height*width
    
    lin_scale=np.sqrt( float(mpixels/TARGET_MPIXELS) )
    if lin_scale<0.9 or lin_scale>1.1 : #avoid rescale if dimensions are close
        new_width=int (width/lin_scale)
        new_height= int(height/lin_scale)
        im_r = cv2.resize(im_r, (new_width, new_height) )
    
    return im_r


def LoadImage(img_name):
    """            
    Function definition
    +++++++++++++++++++
            
        .. py:function:: LoadImage(img_name)

            Loads and filters an image.
            
            :param string img_name: filename of the image or path if necessary.
               
            :rtype: image - three dimensional numpy array or one dimensional if image is grayscale.
    """
    im_r = cv2.imread(img_name)
    im_r = ndimage.median_filter(im_r, 6)
    im_r = ImageRescale(im_r)
    
    return im_r

def SplitImage(img, equalize = False, silence=False):    
    """            
    Function definition
    +++++++++++++++++++
            
        .. py:function:: SplitImage(img)

            Splits an RGB image to channels and print the result.
            
            :param string img: RGB image to be splitted.
            :param boolean silence: default is True. Set to False to print the result.
               
            :rtype: r, g, b - three numpy arrays containing red, green and blue channels of the image.
    """
    b, g, r = cv2.split(img)
    
    if equalize:
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)
     
    
    if silence == False:
        titles = ['Original Image', 'Blue', 'Green', 'Red']
        images = [img, b, g, r ]

        plt.figure()

        for i in xrange(4):
            plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        
        plt.show()

    return r,g,b





    
    
    
    
    
    
    
