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
    
def Closing(img, CLO=2, silence=True):
    closing=img
    for i in range(1,CLO):
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1+i*3,1+i*3))
        closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closing
    
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
     
    
def FeaturesDetection(img, total_mask, LOW=15, HIGH=100, TP_MASK=True, KERNEL=15, EQ=False, silence=True):
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
    tophat = np.array(tophat*total_mask, dtype="uint8") 
    
    if silence==False: ImU.PrintImg(tophat,'tophat image')
    
    ret,thresh = cv2.threshold(tophat,LOW,HIGH,cv2.THRESH_BINARY)
    ImU.PrintImg(thresh,'tophat & threshold')

    
    '''    
    tophat=ImU.ContrastCorrection(tophat,1.5)

    
    if silence==False: ImU.PrintImg(tophat,'tophat mult x image') 
    
    ret,thresh = cv2.threshold(tophat,LOW,HIGH,cv2.THRESH_BINARY)
    ImU.PrintImg(thresh,'tophat mult x & threshold')
    '''
    
    if silence==False: ImU.PrintImg(tophat,'after tophat')
        
    #threshold
   

     
    #thresh= np.array(thresh*total_mask, dtype="uint8")       
      
    #mask = np.ones(thresh.shape[:2], dtype="uint8") * 255  
    


 
    return tophat, thresh


    
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

def MatchedFilter(img):
    '''
    M = cv2.getRotationMatrix2D((cols/2,rows/2),degs,1) #cols/2,rows/2 defines the center of rotation, last argument is scale
    rot_g = cv2.warpAffine(g_original,M,(cols,rows)) # Rotation is done    
    '''
    kernel = np.ones((5,5),np.float32)/25
    kernel_x = np.ndarray( shape=(5,5), dtype="int" )
    kernel_y = np.ndarray( shape=(5,5), dtype="int" )
    
    
    img = cv2.GaussianBlur(img,(9,9),5) 
   
    kernel_x[0] = [-2, -1, 0, 1, +2]
    kernel_x[1] = [-2, -1, 0, 1, +2]
    kernel_x[2] = [-2, -1, 0, 1, +2]
    kernel_x[3] = [-2, -1, 0, 1, +2]
    kernel_x[4] = [-2, -1, 0, 1, +2]       
    
    kernel_y[0] = [-2,-2,-2,-2,-2]
    kernel_y[1] = [-1,-1,-1,-1,-1]
    kernel_y[2] = [ 0, 0, 0, 0, 0]
    kernel_y[3] = [ 1, 1, 1, 1, 1]
    kernel_y[4] = [+2,+2,+2,+2,+2]
    
    pi= math.pi
    #thetas= [0, 0.25*pi, 0.5*pi, 0.75*pi, 1*pi, 1.25*pi, 1.5*pi, 1.75*pi]
    
    thetas= [0, 0.5*pi,  1*pi,  1.5*pi ] 
    
    x,y=img.shape
    dst= np.ndarray( shape=(x,y), dtype="uint8" )    
    responses=list()
    #responses = np.ndarray(shape=(4,x,y) , dtype="uint8")
    #i=0
    for theta in thetas:
        kernel = kernel_x*math.cos(theta) + kernel_y*math.sin(theta)
        
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
            
            for z_pix in range(4):
                if responses[z_pix][x_pix][y_pix]> max_pix: max_pix= responses[z_pix][x_pix][y_pix]
            
            max_responses[x_pix][y_pix]=  max_pix
            max_pix=-1
            
            
    #ret,max_responses = cv2.threshold(max_responses,50,127,cv2.THRESH_BINARY) 
    ImU.PrintImg(max_responses,'max_responses')
    
    return max_responses
          

def MatchedFilter2(img):
    '''
   
    '''
    #kernel = np.ones((5,5),np.float32)/25
    kernel_x = np.ndarray( shape=(5,5), dtype="int" )
    kernel_y = np.ndarray( shape=(5,5), dtype="int" )
    
   
    
    img = cv2.GaussianBlur(img,(31,31),5) 
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
    thetas= [0, 45, 90, 135, 180, 225, 270, 315]#, 45, 60]#, 0.5*pi , 0.75*pi,  1*pi,  1.25*pi , 1.5*pi, 1.75*pi ] 
    
    x,y=img.shape
    dst= np.ndarray( shape=(x,y), dtype="uint8" )
    rot_kernel = np.zeros(shape=(16,16), dtype="int" )
     
    responses=list()
    #responses = np.ndarray(shape=(4,x,y) , dtype="uint8")
    #i=0
    
    kernel=np.uint8(kernel +10)
    rot_kernels=list()
    for theta in thetas:
        ''' 
        R=[ [math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta) ] ] #rotation matrix
        
        R=np.asarray(R)
        
        for xx in range(16):
            for yy in range(16):
                
                a= np.asarray ([ xx, yy  ])
                
                
                a= a+8 # translate
                b = R.dot (a.transpose())
                b= b-8
                if b.all()>=0 and b.all <=16:
                    rot_kernel[b[0]][b[1]]=kernel[xx][yy]
        
   
        ImU.PrintImg(rot_kernel + 15,'rot kernel')
        
        
        '''

        M = cv2.getRotationMatrix2D((8,8),theta,1) #cols/2,rows/2 defines the center of rotation, last argument is scale
        rot_kernel = cv2.warpAffine(kernel,M,(16,16), borderValue=10) # Rotation is done
        #ImU.PrintImg(rot_kernel,'rot kernel') 
        rot_kernel=(rot_kernel.astype(int)-10).astype(int)
        rot_kernel=rot_kernel/2.
        rot_kernels.append(rot_kernel)
        
        dst = cv2.filter2D(img,-1,rot_kernel) #-1 means the same depth as original image         
        responses.append(dst)
              
    # Find max responses
    max_responses = np.zeros( shape=(x,y), dtype="uint8" )
    max_pix=-1
    for x_pix in range(x):
        for y_pix in range(y):
            
            for z_pix in range(len(thetas)):
                if responses[z_pix][x_pix][y_pix]> max_pix: max_pix= responses[z_pix][x_pix][y_pix]
            
            max_responses[x_pix][y_pix]=  max_pix
            max_pix=-1
            
            
    #ret,max_responses = cv2.threshold(max_responses,50,127,cv2.THRESH_BINARY) 
    ImU.PrintImg(max_responses,'max_responses')
      
    return 0#max_responses
        
def DetectHE(img, gamma_offset=0, silence=False):
    
    img=HistAdjust(img, gamma_offset=0, silence=True)
    #img=ImU.GammaCorrection(img,4)

    dilate, closing, opening = BasicMorphology(img, DIL=3, CLO=3, silence=silence) #golden params so far DIL=3, CLO=3 
    circular_mask, fill_mask, circular_inv, total_mask = Msk.CircularDetectMasking(img, opening, silence=True)
    
    x,y= Msk.Disc_Detect(img,'WHITE')
    optic_disc_mask= Msk.DiscMask(circular_mask, x,y,65)
    total_mask= total_mask*optic_disc_mask #*vessels_mask
    
    #opening=255-opening
    # ImU.PrintImg(optic_disc_mask,'optic_disc_mask test')
    tophat = FeaturesDetection(opening, total_mask, LOW=15, HIGH=100, TP_MASK=True, KERNEL=10,EQ=False, silence=True) #default=opening
    #tophat = FeaturesDetection(opening, total_mask, LOW=15, HIGH=100,  EQ=True, silence=True) #default=opening
    #opening=ImU.ContrastCorrection(opening,1) 
    
    tophat = FeaturesDetection(opening, total_mask, LOW=15, HIGH=100, TP_MASK=True, KERNEL=15,EQ=False, silence=True)
    tophat = FeaturesDetection(opening, total_mask, LOW=15, HIGH=100, TP_MASK=True, KERNEL=20,EQ=False, silence=True)
    
    return tophat


def DetectVessels(img, gamma_offset=0, silence=True):
    
    img=HistAdjust(img, gamma_offset=0, silence=True)
    img=ImU.GammaCorrection(img,0.7)
    
    #only for mask
    #dilate, closing, opening = BasicMorphology(img, DIL=3, CLO=4, silence=False)
            
    #circular_mask, fill_mask, circular_inv, total_mask = Msk.CircularDetectMasking(img, opening, silence=False)
    
    erode =  Erode(img, EROD=2, silence=True)    
    
    img=255-erode
    

    #tophat
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
     
    #closing
    #tophat= Closing(tophat, CLO=6, silence=True)
    #kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel, iterations=4)
   
    if silence==False: ImU.PrintImg(tophat,'after tophat')
    #threshold
    ret,thresh = cv2.threshold(tophat,10,50,cv2.THRESH_BINARY)
    
    thresh =1-thresh/thresh.max()
    
    if silence==False: ImU.PrintImg(thresh,'tophat & threshold')    
    
    #tophat, mask2 = FeaturesDetection(img, total_mask, EQ=False,  silence=False)
    return thresh
    

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

   





    
    
    
    
    
    
    
