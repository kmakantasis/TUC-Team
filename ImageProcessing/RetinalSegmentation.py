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
import ImageProcessing as ImP

from numba import double
from numba.decorators import jit

 
def DetectFlow_1(img):

    total_mask=Msk.TotalMask(img)
    vessels = DetectVessels(img,total_mask, ContureFilter=True, silence=True)
    HEs_Grey, HEs_Bin = DetectHE(img, gamma_offset=0, silence=True)

    return 1    
    
        
def DetectVesselsFast(img, silence=True):
    img = cv2.GaussianBlur(img,(3,3),8)
    img = cv2.GaussianBlur(img,(7,7),4)
    #img = cv2.GaussianBlur(img,(15,15),2)
    adaptiveThreshold=cv2.adaptiveThreshold(img,5,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,31,2)
    if silence==False: ImU.PrintImg(adaptiveThreshold,'adaptiveThreshold')
 
    return adaptiveThreshold        

 
        
@jit       
def DetectVessels(img, total_mask, ContureFilter=True, silence=True):
    print "starting Detect Vessels"
    #img=ImU.ImageRescale(img, TARGET_MPIXELS=0.5e6, GRAY=True)
    #kernel = np.ones((5,5),np.float32)/25
    #img=Erode(img, EROD=4)
    
    img = cv2.GaussianBlur(img,(3,3),4)
    img = cv2.GaussianBlur(img,(7,7),2)
    #img = cv2.GaussianBlur(img,(15,15),2)
    
    #img2=ImU.BandCorrection(img,127,255, 0.6)
     
 
    #img=img2
    

    kernel_zero = np.zeros(shape=(1,16), dtype="int")
    kernel_line = np.zeros(shape=(1,16), dtype="int")
    kernel_line = np.array([0, 4, 3, 2, 1, -2, -5, -6, -5, -2, 1 ,2, 3, 4, 0, 0])
    kernel_line.shape=(1,16)
 

    kernel=[kernel_zero,
            kernel_zero,
            kernel_zero,
            kernel_line,
            kernel_line,                    
            kernel_line,
            kernel_line,
            kernel_line,
            kernel_line,
            kernel_line,
            kernel_line,
            kernel_line,
            kernel_line, 
            kernel_zero,
            kernel_zero,
            kernel_zero]   
 
    kernel=np.asarray(kernel).reshape((16,16))
 
    pi=math.pi
    #thetas= [0, 0.25*pi]#, 0.5*pi , 0.75*pi,  1*pi,  1.25*pi , 1.5*pi, 1.75*pi ] 
    #thetas= [0, 45, 90, 135, 180, 225, 270 , 315]#, 45, 60]#, 0.5*pi , 0.75*pi,  1*pi,  1.25*pi , 1.5*pi, 1.75*pi ] 
    #thetas= [0, 15, 30, 45,60 , 75, 90, 105, 120, 135,150, 165, 180]
    thetas= [0, 30,  60,  90, 120, 150, 180]
    
    x,y=img.shape
    dst= np.ndarray( shape=(x,y), dtype="uint8" )
    rot_kernel = np.zeros(shape=(16,16), dtype="int" )
     
    
    #responses = np.ndarray(shape=(4,x,y) , dtype="uint8")
    #i=0
  
    
    kernel=np.uint8(kernel +10)
    rot_kernels=[]
    responses=list()
    for theta in  thetas: #np.arange(0, 180 ,30):
   
        M = cv2.getRotationMatrix2D((8,8),theta,1) #cols/2,rows/2 defines the center of rotation, last argument is scale
        rot_kernel = cv2.warpAffine(kernel,M,(16,16), borderValue=10) # Rotation is done
        #ImU.PrintImg(rot_kernel,'rot kernel') 
        rot_kernel=(rot_kernel.astype(int)-10).astype(int)
        rot_kernel=rot_kernel/8.
        rot_kernels.append(rot_kernel)
        
        dst = cv2.filter2D(img,-1,rot_kernel) #-1 means the same depth as original image         
        responses.append(dst) #append tuple
        
        
    responses=np.asarray(responses)     
    '''   
    max_response = np.zeros_like(img)
    for response in responses:
        np.maximum(max_response, response, max_response)
    '''        
             
    max_response = np.zeros( shape=(x,y), dtype="uint8" )
    max_response_theta = np.zeros( shape=(x,y), dtype="uint8" )
    max_pix=-1
    for x_pix in range(x):
        for y_pix in range(y):  
            for z_pix in range(len(thetas)):
                if responses[z_pix][x_pix][y_pix]> max_pix:
                    max_pix= responses[z_pix][x_pix][y_pix]
                    max_z=z_pix
            
            max_response[x_pix][y_pix]=  max_pix
            max_response_theta[x_pix][y_pix] =  max_z*30
            max_pix=-1
   
           
    #max_response=max_responses[0]
    ret,thresh = cv2.threshold(max_response,30,127,cv2.THRESH_BINARY_INV) 
     
    #max_response=Dilate(max_response, DIL=4, KERNEL=2)
    #max_response=Opening(max_response, OPEN=4) 
    
    #ImU.PrintImg(max_response,'max_responses')  
    #max_responses=Erode(max_responses, EROD=1)
    #ImU.PrintImg(max_response,'Erode/dilate')
 
    #ret, otsu = cv2.threshold( max_response,0,127,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #ImU.PrintImg(otsu,'otsu')
    
    #thresh=cv2.adaptiveThreshold(max_response,1,cv2.ADAPTIVE_THRESH_MEAN_C  , cv2.THRESH_BINARY_INV,31,-15)
    #ImU.PrintImg(adaptiveThreshold,'adaptiveThreshold')
  
    #erode=Erode(adaptiveThreshold,EROD=1, KERNEL=6)
    #erode=Dilate(erode,DIL=1, KERNEL=6)
    #ImU.PrintImg(erode,'Erode adaptiveThreshold') 
    
    if ContureFilter==True: 
        final_vessels_mask=CntP.VesselsFiltering(thresh)
    else:
        final_vessels_mask = thresh
    
    #---------------------------------masking--------------------
    #total_mask=Msk.TotalMask(img)  
    #------------------------------------------------------------
    
    final_vessels_mask= (1- final_vessels_mask/final_vessels_mask.max())*total_mask
    
    if silence==False: ImU.PrintImg(final_vessels_mask,' final_vessels_mask & mask')    
    
    skel=ImP.Skeletonize(final_vessels_mask)
    
    if silence==False: ImU.PrintImg(skel,'skeletonize')
    #cnt_filtered=CntP.VesselsFiltering(cnt_filtered)
    #ImU.PrintImg(cnt_filtered,'cnt_filtered') 
      
    return final_vessels_mask#otsu/otsu.max()
        
def DetectHE(img, gamma_offset=0, silence=False):
    
    img=ImU.HistAdjust(img, gamma_offset=0, silence=True)
    #img=ImU.GammaCorrection(img,4)

    dilate, closing, opening = ImP.BasicMorphology(img, DIL=3, CLO=3, silence=silence) #golden params so far DIL=3, CLO=3 
    circular_mask, fill_mask, circular_inv, total_mask = Msk.CircularDetectMasking(img, opening, silence=True)
 
    #ImU.PrintImg(total_mask,'circular_mask')
    simple_mask_cirlualr=Msk.CircularMaskSimple(img)
    x,y= Msk.Disc_Detect(img,'WHITE')
    optic_disc_mask= Msk.DiscMask(img, x,y,70)
    
    #ImU.PrintImg(optic_disc_mask,'optic_disc_mask')
    
    vessels_mask= DetectVesselsFast(img)/DetectVesselsFast(img).max()
    if silence==False: ImU.PrintImg(vessels_mask,'vessels_mask')
    
    total_mask= optic_disc_mask*vessels_mask*simple_mask_cirlualr #*total_mask*
    total_mask=total_mask/total_mask.max()
    #ImU.PrintImg(total_mask,'total_mask')
    
    #opening=255-opening
    # ImU.PrintImg(optic_disc_mask,'optic_disc_mask test')
 
    tophat, thresh = ImP.FeaturesDetection(opening, total_mask, LOW=15, TP_MASK=True, KERNEL=10,EQ=False, silence=True) #default=opening
    #tophat = FeaturesDetection(opening, total_mask, LOW=15, HIGH=100,  EQ=True, silence=True) #default=opening
    #opening=ImU.ContrastCorrection(opening,1) 
    #if silence==False: ImU.PrintImg(img ,'img')
 
        
    #ret,thresh = cv2.threshold(erode,60,127,cv2.THRESH_BINARY)
    #ImU.PrintImg(thresh,'erode & thresh')
    
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #gradient = cv2.morphologyEx(erode, cv2.MORPH_BLACKHAT, kernel)

    if silence==False: ImU.PrintImg(tophat,'HE out')
    
    return tophat, thresh



   





    
    
    
    
    
    
    
