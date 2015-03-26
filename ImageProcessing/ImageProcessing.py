# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import math
import cv2

import ContourProcessing as CntP
import ImageUtils as ImU
import MaskingUtils as Msk


def Dilate(img, DIL=5, CLO=4, silence=True):
    dilate=img
    for i in range(1,DIL):
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1+i,i+1))
        dilate  = cv2.dilate(dilate,kernel,iterations = 1)
        
    return dilate
    
def Erode(img, EROD=2, silence=True):
  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    erode = cv2.erode(img,kernel,iterations = EROD)
        
    return erode    
    
    
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
     
    
def FeaturesDetection(img, total_mask,  TP_MASK=True, EQ=False, silence=True):
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
    if EQ:
        img = cv2.equalizeHist(img)
        
    if silence==False: ImU.PrintImg(img,'before tophat')
       
        
    #tophat
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
  

    if silence==False: ImU.PrintImg(tophat,'after tophat')
    #threshold
    ret,thresh = cv2.threshold(tophat,25,100,cv2.THRESH_BINARY)
    
    if silence==False: ImU.PrintImg(thresh,'tophat & threshold')
     
    #thresh_test
    #ret, thresh_test = cv2.threshold( img,0,253,cv2.THRESH_BINARY) 
    
     
    if TP_MASK==True:
        thresh= np.array(thresh*total_mask, dtype="uint8")       
    
    
    thresh2=thresh+0 #otherwise it affects thresh
    mask = np.ones(thresh.shape[:2], dtype="uint8") * 255
    cnt = cv2.findContours(thresh2,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    

    cv2.drawContours(mask, cnt, -1, (0,50,100), 2)    
    
    # We sort and discard possible noisy features/artifacts
    img_height, img_width = img.shape
    mask2 = np.ones(thresh.shape[:2], dtype="uint8") * 255

    
    for c in cnt :
          
        #-----we should put our rules here
        Rules_Passed=False
        
        rule0 = CntP.CNTRule_Area(c, 10, 1200)
        rule1 = CntP.CNTRule_KickOutCircular(img,c)
        rule2 =1# CntP.CNTRule_Sphericity(c)
        rule3 = CntP.CNTRule_AspectRatio(c)
        
        Rules_Passed= rule0 and rule1 and rule2 and rule3
            
        if Rules_Passed : #kick out very large artifacts
            cv2.drawContours(mask2, [c], -1, 0, -1)
                             
            
    #quality_percent = float(quality_meter)/ (len(cnt)+1)
   # quality_mass_percent  =  float(quality_mass)/ (total_mass+1)


    mask2=1- mask2/mask2.max()  
    tophat= tophat*mask2 
    
    if silence==False: ImU.PrintImg(tophat,'contour filtered image')
 
        
    if silence==True:  
        titles = ['Refined Contour mask', 'Refined Tophat']
        images = [mask2, tophat]

        plt.figure() 
        
        for i in xrange(2):
            plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        
        plt.show()
       
        
    return tophat, mask2
    
def HistAdjust(img, gamma_offset=0, silence=True):
    hist = cv2.calcHist([img],[0],None,[4],[0,256])
    if silence==False:  
        print ("Hist[0]=%3.3f" %hist[0])
    #----histogram correction invariant to scale
    height, width = img.shape
    mpixels=height*width
    print ("Hist mpixels=%3.3f" %mpixels)
    
          
    if (hist[0]<mpixels/2.):
        gamma= abs(0.55*mpixels-hist[0])/(0.2*mpixels) +1 + gamma_offset
        img= ImU.GammaCorrection(img,gamma)
    else:
        gamma=1 + gamma_offset
        img= ImU.GammaCorrection(img,gamma)
        
    hist = cv2.calcHist([img],[0],None,[4],[0,256])
    if silence==False:  
        print ("After Gamma=%2.2f Hist[0]=%3.3f" %(gamma,hist[0]) )  
        print ("channel mean=%3.3f" %np.mean(img))
        
    return img
        
        
def DetectHE(img, gamma_offset=0, silence=True):
    
    img=HistAdjust(img, gamma_offset=0, silence=True)
    
    dilate, closing, opening = BasicMorphology(img, DIL=3, CLO=4, silence=silence)
            
    circular_mask, fill_mask, circular_inv, total_mask = Msk.CircularDetectMasking(img, opening, silence=silence)
    
    
    #tophat, mask2 = FeaturesDetection(img, total_mask, EQ=False, silence=False) #default=opening
    mask2=0
    tophat=DetectVessels(img, gamma_offset=0, silence=False)
    
    return tophat, mask2



def DetectVessels(img, gamma_offset=0, silence=True):
    
    img=HistAdjust(img, gamma_offset=0, silence=True)
     
    dilate, closing, opening = BasicMorphology(img, DIL=3, CLO=4, silence=silence)
            
    circular_mask, fill_mask, circular_inv, total_mask = Msk.CircularDetectMasking(img, opening, silence=silence)
    
    
    erode =  Erode(img, EROD=2, silence=True)    
    
    img=255-erode
    
    #tophat
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
  

    if silence==False: ImU.PrintImg(tophat,'after tophat')
    #threshold
    ret,thresh = cv2.threshold(tophat,25,200,cv2.THRESH_BINARY)
    
    if silence==False: ImU.PrintImg(thresh,'tophat & threshold')    
    
 
    #tophat, mask2 = FeaturesDetection(img, total_mask, EQ=False,  silence=False)
    return tophat
    

def CropImage(img, features, silence=True):
    ret,thresh_flip = cv2.threshold(img,10,1,cv2.THRESH_BINARY)  

    contours,hierarchy = cv2.findContours(thresh_flip, 1, 2)
    
    area = 0
    for cnt in contours:
        if area < cv2.contourArea(cnt):
            area = cv2.contourArea(cnt)
            largest_contour = cnt
    
    cnt = largest_contour
    x,y,w,h = cv2.boundingRect(cnt)
   
    if w > h:
        max_dim = w
    else:
        max_dim = h

    cropped_img = np.zeros((max_dim, max_dim), dtype='uint8')
    cropped_img[0:h, 0:w] = features[y:y+h,x:x+w]
    
    if silence == False:
        plt.figure()
        plt.imshow(cropped_img ,cmap = 'gray')    
        plt.show()
    
    return cropped_img

   





    
    
    
    
    
    
    
