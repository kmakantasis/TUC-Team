# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
 
 
def PrintImg(im_r, meassage):

    plt.figure()
    plt.title(meassage)
    plt.axis('off')
    plt.imshow(im_r, cmap = 'gray')
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

def ContrastCorrection(img, correction):

    x,y=img.shape
    bright= np.ndarray( shape=(x,y), dtype="uint8" )
    bright.fill(2)  
    img = cv2.multiply(img, bright)
    return img
    
    
def DetectMicroAN(img, EROD=4, CLO=4, OPEN=5, silence=False):
    #Under heavy development
    '''
    hist = cv2.calcHist([img],[0],None,[4],[0,256])
    if silence==False:  
        print ("Hist[0]=%3.3f" %hist[0])
    #----histogram correction invariant to scale
    height, width = img.shape
    mpixels=height*width
    print ("Hist mpixels=%3.3f" %mpixels)
 
        
    if (hist[0]<mpixels/2.):
        gamma= abs(0.55*mpixels-hist[0])/(0.2*mpixels) +1
        img=ImU.GammaCorrection(img,gamma)
    else:
        gamma=1
    '''    
    ###-------------basic morphology
    
        
    erode=closing=dilate=img
    #Basic morphological operations
    
    #erode
    for i in range(1,EROD):
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        erode =cv2.erode(img,kernel,iterations = 1)
   
    erode=255-erode
    
    plt.figure()
    plt.title("MicroAN erode")
    plt.imshow(erode, cmap = 'gray')
    plt.show()     
        
    '''   
    #blackhat
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    erode = cv2.morphologyEx(erode, cv2.MORPH_BLACKHAT, kernel)
    plt.figure()
    plt.title("MicroAN Blackhat")
    plt.imshow(erode, cmap = 'gray')
    plt.show()       
    '''
  
        
    '''    
    #otsu
    ret, erode = cv2.threshold( erode,40,127,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    plt.figure()
    plt.title('Otshu Image')
    plt.imshow(erode ,cmap = 'gray')
    plt.show()  
    '''
        

     
    #Gaussian filter
    ''' 
    for i in range(1,4):
        erode = cv2.GaussianBlur(erode,(2*i+1,2*i+1),3) 
    ''' 


    ''' 
    #laplacian filter
    #erode = cv2.Laplacian(erode,cv2.CV_64F)
    sobelx = cv2.Sobel(erode,cv2.CV_64F,1,0,ksize=5)
    
    plt.figure()
    plt.title("sobelx")
    plt.imshow(sobelx, cmap = 'gray')
    plt.show() 
    
    sobely = cv2.Sobel(erode,cv2.CV_64F,0,1,ksize=5)
    
    plt.figure()
    plt.title("sobely")
    plt.imshow(sobely, cmap = 'gray')
    plt.show()     
    '''
    
    #canny edge
    edges = cv2.Canny(erode,170,170)    
    plt.figure()
    plt.title("Canny edges")
    plt.imshow(edges, cmap = 'gray')
    plt.show()
    
    
  
    #Gradient filter
    ''' 
    for i in range(1,2):
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        erode = cv2.morphologyEx(erode, cv2.MORPH_GRADIENT, kernel)
    '''
    #erode=cv2.equalizeHist(erode)
    '''
    # Tophat
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(120,120))
    erode = cv2.morphologyEx(erode, cv2.MORPH_TOPHAT, kernel)
    '''
    #ret,erode = cv2.threshold(erode,100,127,cv2.THRESH_BINARY)  

 
    '''   
    #closing  
    closing=dilate
    for i in range(1,CLO):
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1+i*3,1+i*3))
        closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel, iterations=1)
    ''' 

               
    #opening
    for i in range(1,OPEN):
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel)
    
   
    plt.figure()
    plt.title("MicroAN openig")
    plt.imshow(opening, cmap = 'gray')
    plt.show()    
        
    ##- end basic morphology
       
   
            
    #circular_mask, fill_mask, circular_inv, total_mask = CircularDetectMasking(img, opening, silence=silence)

    #tophat, mask2 = FeaturesDetection(opening, total_mask, silence=silence) 
    
    return 1


    
    
    
    
    
    
    
