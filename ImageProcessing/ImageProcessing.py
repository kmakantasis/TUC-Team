# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import math
import cv2
from scipy.ndimage import label
from scipy import ndimage
from skimage.morphology import watershed
from skimage.morphology import skeletonize
import operator

import ContourProcessing as CntP
import ImageUtils as ImU
import MaskingUtils as Msk

from numba import double
from numba.decorators import jit


def Dilate(img, DIL=2, KERNEL=3):
    dilate=img
    for i in range(1,DIL):
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(KERNEL,KERNEL))
        dilate  = cv2.dilate(dilate,kernel,iterations = DIL)
        
    return dilate
    
def Erode(img, EROD=2, KERNEL=3):
  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(KERNEL,KERNEL))
    erode = cv2.erode(img,kernel,iterations = EROD)
        
    return erode
    
def Closing(img, CLO=2, silence=True):
    closing=img
    for i in range(1,CLO):
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1+i*3,1+i*3))
        closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closing
    
def Opening(img, OPEN=2, silence=True):   
    for i in range(1,OPEN):
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1+i,1+i))
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening
    
    
def BasicMorphology(img, DIL=5, CLO=4, silence=True):
    """            
    Function definition
    +++++++++++++++++++
            
        .. py:function:: BasicMorphology(img, DIL=5)

            Apply basic morphology operations ( dilate, closing, opening) on input image.
            
            :param uint8 img: grayscale image to be processed.
            :param int DIL: default=5, iterations of dilating operation.
            :param int COL: default=4, iterations of closing operation.
            :param boolean silence: default is True. Set to False to print the result.
               
            :rtype: dilate, closing, opening - thre two dimensional numpy arrays corresponding to processed images. 
    """
    closing=dilate=img
    #Basic morphological operations
    #Dilate: eliminates vessels
    for i in range(1,DIL):
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1+i,i+1))
        dilate = cv2.dilate(dilate,kernel,iterations = 1)
    
         
    #closing  
    closing=dilate
    for i in range(1,CLO):
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1+i*3,1+i*3))
        closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel, iterations=1)
                 
    #opening
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
         
    if silence == False:
        titles = ['Original Image', 'Dilated', 'Opened', 'Closed']
        images = [img, dilate, closing, opening ]

        plt.figure()        
        
        for i in xrange(4):
            plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        
        plt.show()
    
    return dilate, opening, closing
     
    
def FeaturesDetection(img, total_mask, LOW=15, TP_MASK=True, KERNEL=15, EQ=False, silence=True):
    """            
    Function definition
    +++++++++++++++++++
            
        .. py:function:: FeaturesDetection(opening, total_mask, TP_MASK=True, silence=True)
        
            Functions for detecting features.
            
            :param np.array opening: opened image created using BasicMorphology().
            :param np.array total_mask: opened image created using CircuralDetectMasking().
            :param boolean TP_MASK: default is True. apply mask later, after thresholding. 
            :param boolean silence: default is True. Set to False to print the result.
            
               
            :rtype: tophat, mask2 - two two dimensional numpy arrays corresponding to features. 
    """
    if silence==False: ImU.PrintImg(total_mask,'total mask')    
    
    if EQ: img = cv2.equalizeHist(img)
        
    if silence==False: ImU.PrintImg(img,'before tophat')
       
     
    #tophat
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(KERNEL,KERNEL))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
  
    #apply mask
    tophat = np.array(tophat*(total_mask/total_mask.max()), dtype="uint8") 
    
    if silence==False: ImU.PrintImg(tophat,'tophat image')
    
    ret,thresh = cv2.threshold(tophat,LOW,1,cv2.THRESH_BINARY)
    if silence==False:  ImU.PrintImg(thresh,'tophat & threshold')

    
    '''
    #-------------------Contrast correction... Very Beta ---------------------
    tophat=ImU.ContrastCorrection(tophat,1.5)

    
    if silence==False: ImU.PrintImg(tophat,'tophat mult x image') 
    
    ret,thresh = cv2.threshold(tophat,LOW,HIGH,cv2.THRESH_BINARY)
    ImU.PrintImg(thresh,'tophat mult x & threshold')
    '''
    
    if silence==False: ImU.PrintImg(tophat,'after tophat')
        
   
    return tophat, thresh

    
def BuildGaborFlters():
    #https://cvtuts.wordpress.com/2014/04/27/gabor-filters-a-practical-overview/

    filters = [] #init list
    ksize = 17
    for theta in np.arange(0, np.pi, np.pi / 8):
        kern = cv2.getGaborKernel((ksize, ksize), 3, theta, 8.0, 0.8, 0, ktype=cv2.CV_32F)
        kern /= 1.0*kern.sum()
        filters.append(kern)
    return filters

def ProcessGabor(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum    
    

def MatchedFilter(img):
    '''
    M = cv2.getRotationMatrix2D((cols/2,rows/2),degs,1) #cols/2,rows/2 defines the center of rotation, last argument is scale
    rot_g = cv2.warpAffine(g_original,M,(cols,rows)) # Rotation is done    
    
    '''
    #---------------gabor filter testing
    ImU.PrintImg(img,'img ')
    filters= BuildGaborFlters()
    gabor_filtered= ProcessGabor(img, filters)
    #gabor_filtered=Closing(gabor_filtered)
    #gabor_filtered= ProcessGabor(gabor_filtered, filters) 
    ImU.PrintImg(gabor_filtered,'gabor_filtered ')    
    
    #hist_eq=cv2.equalizeHist(gabor_filtered)
    
    #ImU.PrintImg(hist_eq,'hist_eq gabor_filtered ')
    
    #ret,thresh = cv2.threshold(hist_eq,15,50,cv2.THRESH_BINARY)
    #ImU.PrintImg(thresh,'hist_eq & threshold')

    #adaptive thresh
    #adaptiveThreshold=cv2.adaptiveThreshold(gabor_filtered,5,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,31,2)
    #ImU.PrintImg(adaptiveThreshold,'gabor_filtered adaptiveThreshold ')    
    
    #otsu
    #ret, otsu = cv2.threshold( gabor_filtered,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #ImU.PrintImg(otsu,'max_responses otsu')
    return 0
    #--------end gabor filter testing  
    
    x,y=img.shape

    img = cv2.GaussianBlur(img,(3,3),8) 
    img = cv2.GaussianBlur(img,(7,7),4) 
    
    
    #kernel_line =np.ndarray( shape=(1,5), dtype="int" )
    #kernel_line = np.asarray( [-1, -0, -0, 0, 0, 0, +1] ) #perwitt
    kernel_line = np.asarray( [-2, 0, 0,  0, +2] ) #perwitt
    #kernel_line = np.asarray( [-1, -2, -3, -2, -1, 2, +1] ) #sobel
    
    smoothing_line  = np.asarray([ 1, 4, 6, 4, 1])
    gradient_line= np.asarray([ 1, 2, 0,-2,-1])
    
    sobel_x= np.multiply.outer (smoothing_line,gradient_line)
    sobel_y= np.multiply.outer (gradient_line,smoothing_line)
    
    KERNEL_SIZE =len(kernel_line)
    
    kernel_x  = [
            kernel_line,
            kernel_line,
            kernel_line,
            kernel_line,
            kernel_line ]
   
    kernel_x=np.asarray(kernel_x).reshape((KERNEL_SIZE,KERNEL_SIZE))
    
    kernel_y = [
            kernel_line.transpose(),
            kernel_line.transpose(),
            kernel_line.transpose(),
            kernel_line.transpose(),
            kernel_line.transpose()  ]

    kernel_y=np.asarray(kernel_y).reshape((KERNEL_SIZE,KERNEL_SIZE)) 
    
    pi= math.pi
    #thetas= [0, 0.25*pi, 0.5*pi, 0.75*pi, 1*pi]
    #thetas= np.asarray([0,  0.125 ,  0.25 , 0.375 , 0.5 , 0.625, 0.75, 1 ] )
    thetas= np.asarray([0, 0.5, 1 ] )
    thetas=thetas*pi
    
    dst= np.ndarray( shape=(x,y), dtype="uint8" )    
    responses=list()
    #responses = np.ndarray(shape=(4,x,y) , dtype="uint8")
    #i=0
    for theta in thetas:
        #kernel = kernel_x*math.cos(theta) + kernel_y*math.sin(theta)
        kernel = sobel_x*math.cos(theta) + sobel_y*math.sin(theta)/10.
        
        dst = cv2.filter2D(img,-1,kernel) #-1 means the same depth as original image
      
       # responses[i]=dst
        #i=i+1        
        responses.append(dst)
        #ret,dst = cv2.threshold(dst,0,127,cv2.THRESH_BINARY)     
        
        #ImU.PrintImg(img,'original')
        #ImU.PrintImg(dst,'filtered')

    max_responses = np.zeros( shape=(x,y), dtype="uint8" )
    max_pix=-1
    for x_pix in range(x):
        for y_pix in range(y):
            
            for z_pix in range(len(thetas)):
                if responses[z_pix][x_pix][y_pix]> max_pix: max_pix= responses[z_pix][x_pix][y_pix]
            
            max_responses[x_pix][y_pix]=  max_pix
            max_pix=-1

    #
    ImU.PrintImg(max_responses,'max_responses ')
      
    blend=cv2.add(max_responses,img)
    ImU.PrintImg(blend ,'img +max_responses ')

    #ret, otsu = cv2.threshold( blend,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #ImU.PrintImg(otsu,'max_responses otsu')
  
    return max_responses
##------------------------------------------------------very experimental zone
 
         
def find_circles(img):
    
    #img = cv2.equalizeHist(img)    
    #ret,otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #ImU.PrintImg(otsu,'THRESH_OTSU')
    
    params = cv2.SimpleBlobDetector_Params()
 
    # Change thresholds
    params.minThreshold = 30;
    params.maxThreshold = 100;
   
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.6   
    
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.6
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 300
    params.maxArea = 1200
      
    detector = cv2.SimpleBlobDetector(params)
     
    # Detect blobs.
    keypoints = detector.detect(img)
     
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    ImU.PrintImg(im_with_keypoints,'im_with_keypoints')
          
    img = cv2.bilateralFilter(img, 11, 17, 17)
 
    canny = cv2.Canny(img, 30, 60,  9) 
    ImU.PrintImg(canny,'canny')
    
    laplacian = abs( cv2.Laplacian(img,cv2.CV_32F,ksize=9))
    ImU.PrintImg(laplacian,'laplacian')
       
    sobel = abs(cv2.Sobel(img,cv2.CV_32F,1,1,ksize=31) )
    ImU.PrintImg(sobel,'sobel')
    
    return img


def Skeletonize(img):
 
    img2=img/img.max()
    img2 = skeletonize(img2) 
     
    return img2
 



   





    
    
    
    
    
    
    
