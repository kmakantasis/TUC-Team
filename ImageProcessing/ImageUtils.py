# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
import math
import MaskingUtils as Msk
import ImageUtils as ImU
from numba import double
from numba.decorators import jit
import copy
 
 
def PrintImg(im_r, meassage):

    plt.figure()
    plt.title(meassage)
    plt.axis('off')
    plt.imshow(im_r, cmap = 'gray')
    plt.show()

def PrintImgColor(im_BGR, meassage):
    im_RGB=cv2.cvtColor(im_BGR,cv2.COLOR_BGR2RGB) #channel rotation
    
    plt.figure()
    plt.title(meassage)
    plt.axis('off')
    plt.imshow(im_RGB, cmap = 'gray')
    plt.show()
    

def ImageRescale(im_r, TARGET_MPIXELS=1e6, GRAY=False):   
    if GRAY==False:
        height, width, depth = im_r.shape
    else: 
        height, width = im_r.shape
        
    
    mpixels=height*width
    
    lin_scale=np.sqrt( float(mpixels/TARGET_MPIXELS) )
    if lin_scale<0.9 or lin_scale>1.1 : #avoid rescale if dimensions are close
        new_width=int (width/lin_scale)
        new_height= int(height/lin_scale)
        im_r = cv2.resize(im_r, (new_width, new_height) )
    
    #--crop image
    height, width, depth = im_r.shape
    crop_size = int (abs(width - height)/2)
    cropped_im_r = im_r[ 0:height-1, crop_size:width-crop_size-1]
    
    if GRAY==False: 
        return  cropped_im_r
    else:
        return np.uint8(cropped_im_r)


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
    #im_r = ndimage.median_filter(im_r, 6)
    im_r = ImageRescale(im_r)
    
    return im_r

   
    
def ExtractPatch_W(img, radius,silence=False): 
    x,y= Msk.Disc_Detect(img,'WHITE')
    cropped_img = img[ y-radius:y+radius, x-radius:x+radius]
    
    if silence == False: 
        ImU.PrintImg(cropped_img,'cropped_img')
 
    return cropped_img
    
def ExtractPatch_B(img, radius,silence=False): 
    x,y= Msk.Disc_Detect(img,'DARK')
    cropped_img = img[ y-radius:y+radius, x-radius:x+radius]
    
    if silence == False: 
        ImU.PrintImg(cropped_img,'cropped_img')
        
    return cropped_img
    

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


def GammaCorrection(img, correction):
    """            
    Function definition
    +++++++++++++++++++
            
        .. py:function:: GammaCorrection(img, correction)

            Apply gamma correction on input image.
            
            :param uint8 img: grayscale image to be gamma corrected.
            :param float correction: gamma value.
               
            :rtype: uint8 img - two dimensional uint8 numpy array corresponding to gamma corrected image. 
    """
    img = img/255.0
    img = cv2.pow(img, correction)
    return np.uint8(img*255)
    
def HistAdjust(img, gamma_offset=0, silence=True):
    height, width = img.shape
    mpixels=height*width
    hist = cv2.calcHist([img],[0],None,[4],[0,256])
    if silence==False:
        print ("\nchannel mean=%3.3f" %np.mean(img))
        print ("Hist Bands [0]=%3.3f" % (hist[0]/mpixels) )
        print ("Hist Bands [1]=%3.3f" % (hist[1]/mpixels) )
        print ("Hist Bands [2]=%3.3f" % (hist[2]/mpixels) )
        print ("Hist Bands [3]=%3.3f" % (hist[3]/mpixels) )
                
    #----histogram correction invariant to scale

          
    if (hist[0]<mpixels/2.):
        gamma= abs(0.55*mpixels-hist[0])/(0.2*mpixels) +1 + gamma_offset
        img= GammaCorrection(img,gamma)
    else:
        gamma=1 + gamma_offset
        img= GammaCorrection(img,gamma)
        
    hist = cv2.calcHist([img],[0],None,[4],[0,256])
    if silence==False:  
        print ("\nAfter Gamma=%2.2f " %gamma )  
        print ("channel mean=%3.3f" %np.mean(img))
        print ("Hist Bands [0]=%3.3f" % (hist[0]/mpixels) )
        print ("Hist Bands [1]=%3.3f" % (hist[1]/mpixels) )
        print ("Hist Bands [2]=%3.3f" % (hist[2]/mpixels) )
        print ("Hist Bands [3]=%3.3f" % (hist[3]/mpixels) )        
    return img
       
    
@jit  
def BandCorrection(img, A=127, B=255, factor=0.5):
    width,heght=img.shape
    img = np.uint8(img)
    img2=copy.copy(img) #do this or you destroy input object
    for x in range(width):
            for y in range(heght):
                pix=img2[x][y]
                if A<pix:
                    delta =factor*(pix-A)
                    pix = pix-delta 
                    
                if B<pix:
                    delta =factor*(pix-B)
                    pix =pix-delta 
               #else:
                    
                img2[x][y]=pix
    
    return np.uint8(img2)
     
def AverageHue(img):
    width,height, depth=img.shape
 
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_sum= hsv_img[0].sum()
    h_avg=  float(h_sum) /(width*height)
    
    print('h_sum=%.4f' % h_sum)
    print('h_avg=%.4f' % h_avg)
    return   h_avg 

def AverageIntensity(img):#for grey images only
    width,height=img.shape
   
    print ("channel mean=%3.3f" %np.mean(img))

    return   np.mean     

def kmeans(image, segments=8):
       #Preprocessing step
       image=cv2.GaussianBlur(image,(7,7),0)
       vectorized=image.reshape(-1,3)
       vectorized=np.float32(vectorized)
       criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10, 1.0)
       ret,label,center=cv2.kmeans(vectorized,segments,criteria,5,cv2.KMEANS_RANDOM_CENTERS)
       
       center = np.uint8(center)
       res = center[label.flatten()]
       segmented_image = res.reshape((image.shape))
       
       segmented_image=segmented_image
       return label.reshape((image.shape[0],image.shape[1])), segmented_image.astype(np.uint8)

def ContrastCorrection(img, correction):

    x,y=img.shape
    bright= np.ndarray( shape=(x,y), dtype="uint8" )
    bright.fill(2)  
    img = cv2.multiply(img, bright)
    return img
    
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
    



    
    
    
    
    
    
    
