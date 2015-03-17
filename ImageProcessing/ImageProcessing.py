# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import math
import cv2

def ImageRescale(im_r):   
    height, width, depth = im_r.shape
    mpixels=height*width
    TARGET_MPIXELS=1e6
    
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

def SplitImage(img, silence=True):    
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


def Disc_Detect(img2,template,silent=False):

    '''
    Last Maintenance: Antonis
    
    Function definition
    +++++++++++++++++++
            
        .. py:function:: Disc_Detect(image_channel, template)

            Detects circular discs on input image.
            
            :param uint image_channel:  
                           
            :rtype: center_x, 
            :rtype: center_y, 
    ''' 
    
    #Antonis
    
    # All the 6 methods for comparison in a list
    #template = array(Image.open( 'disc_template.jpg' ).convert("L"))
 
    #w, h = template.shape[::-1]
    
    #theight, twidth, tdepth = template.shape() 
    
    template = cv2.resize(template, (180, 180) ) 
    
    w, h = template.shape[::-1]
    methods = ["cv2.TM_CCOEFF_NORMED"]
   # Different methods to choose from
   # , 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
   # methods = ['cv2.TM_SQDIFF_NORMED','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF']
    for meth in methods:
        img = img2.copy()
        method = eval(meth)
    
        # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        center_x= (bottom_right[0]+top_left[0])/2
        center_y= (bottom_right[1]+top_left[1])/2
        if silent==False:
            cv2.rectangle(img,top_left, bottom_right, 0, 14)
            cv2.circle(img,(int(center_x),int(center_y)),10,(255,255,255),-11) 

            plt.subplot(121),plt.imshow(res,cmap = 'gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(img,cmap = 'gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])             
            plt.show()   
            print ("Disc x=%d , y=%d ")  %(center_x,center_y)
            
            
        return  center_x, center_y 
    

def Rotation_Correct(r,g):
    
    '''
    Last Maintenance: Antonis
    Function definition
    +++++++++++++++++++
            
        .. py:function:: Rotation_Correct(red_channel, green_channel)

            Apply gamma correction on input image.
            
            :param uint red_channel:  
            :param uint green_channel:  
               
            :rtype: img - two dimensional numpy array corresponding to rotation corrected image. 
    '''   

    disc_template = cv2.imread('disc_template.jpg',0) 
    x1, y1 = Disc_Detect(r,disc_template)
    
    fovea_template = cv2.imread('fovea_template.jpg',0) 
    x2, y2 = Disc_Detect(g,fovea_template)
    
    dx= abs(x1-x2)
    dy= abs(y1-y2)
    
    print ("Disc1 x=%d , y=%d ")  %(x1,y1)
    
    rads = math.atan2(dy,dx)
    degs = math.degrees(rads)
    
    print ("Fovea-Optic Disc Angle=%2.4f )" %degs)
    
    rows,cols = g.shape
    
    M = cv2.getRotationMatrix2D((cols/2,rows/2),degs,1)
    rot_g = cv2.warpAffine(g,M,(cols,rows))
    plt.imshow(rot_g,cmap = 'gray')
    plt.title('Rotation correct'), plt.xticks([]), plt.yticks([])
    plt.show()
    return(rot_g)  #return green channel rotated


def GammaCorrection(img, correction):
    """            
    Function definition
    +++++++++++++++++++
            
        .. py:function:: GammaCorrection(img, correction)

            Apply gamma correction on input image.
            
            :param string img: RGB image to be splitted.
            :param float correction: gamma value.
               
            :rtype: img - two dimensional numpy array corresponding to gamma corrected image. 
    """
    img = img/255.0
    img = cv2.pow(img, correction)
    return np.uint8(img*255)
    
    
def BasicMorphology(img, DIL=5, CLO=4, silence=True):
    """            
    Function definition
    +++++++++++++++++++
            
        .. py:function:: BasicMorphology(img, DIL=5)

            Apply basic morphology operations ( dilate, closing, opening) on input image.
            
            :param string img: RGB image to be splitted.
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
        dilate =cv2.dilate(dilate,kernel,iterations = 1)
     
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
    
 
def CircularDetectMasking(img, opening, silence=True):
    """            
    Function definition
    +++++++++++++++++++
            
        .. py:function:: CircularDetectMasking(opening)
        
            Creates circular mask and applies masking to discard artifacts in peripheral
            
            :param np.array img: original image.
            :param np.array opening: opened image created using BasicMorphology().
            :param boolean silence: default is True. Set to False to print the result.
            
               
            :rtype: circular, fill_mask, circular_inv, total_mask - four two dimensional numpy arrays corresponding to masks. 
    """
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    dilate=cv2.dilate(img,kernel,iterations = 1) 
    ret,thresh = cv2.threshold(dilate,20,80,cv2.THRESH_BINARY)
    
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))
    erode_thres=cv2.erode(thresh,kernel,iterations = 4)
     
    circular =255-(thresh-erode_thres)
    circular_inv=(thresh-erode_thres)
    #Mask GammaCorrection...maybe median could work
    open_for_mask = GammaCorrection(opening,3)#cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
    fill_mask =   ndimage.binary_fill_holes(open_for_mask)
    #fill_mask  =fill/fill.circular.max()
    circular_mask = circular/circular.max()  #convert to binary  
    
    total  =  (fill_mask + circular_inv)*255
    
    for i in range(1,2):#fills tiny holes in the mask
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*i,2*i))
        total = cv2.morphologyEx(total, cv2.MORPH_OPEN, kernel)
    total_mask=total/total.max()
    
    if silence==False:   
        titles = ['Circular Mask', 'Fill Mask', 'Circular Inv Mask', 'Total  Mask']
        images = [circular_mask, fill_mask, circular_inv, total_mask ]

        plt.figure() 
        
        for i in xrange(4):
            plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        
        plt.show()
        
        
    return circular_mask, fill_mask, circular_inv, total_mask
    
    
def FeaturesDetection(opening, total_mask, TP_MASK=True, silence=True):
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(80,80))
    tophat = cv2.morphologyEx(opening, cv2.MORPH_TOPHAT, kernel)
    
    ret,thresh = cv2.threshold(tophat,30,100,cv2.THRESH_BINARY)  
    
    if TP_MASK==True:
        thresh= np.array(thresh*total_mask, dtype="uint8")       
    
    
    thresh2=thresh+0 #otherwise it affects thresh
    mask = np.ones(thresh.shape[:2], dtype="uint8") * 255
    cnt = cv2.findContours(thresh2,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    cv2.drawContours(mask, cnt, -1, (0,50,100), 2)    
    
    # We sort and discard possible noisy features/artifacts
    mask2 = np.ones(thresh.shape[:2], dtype="uint8") * 255
    quality_meter=0
    total_mass=0
    quality_mass=0
    #-----------------parameters----------------
    AREA_REJECT_A=1200
    AREA_REJECT_B=500
    ASPECT_RATIO=3
    #----------------end parameters-------------
    
    for c in cnt :
        x,y,w,h = cv2.boundingRect(c)
        aspect_ratio = float(w)/h
        total_mass= total_mass + cv2.contourArea(c)
        # if the contour is not bad, draw it on the mask
        if cv2.contourArea(c)<AREA_REJECT_A: #kick out very large artifacts
            if cv2.contourArea(c)>AREA_REJECT_B: #only for large artifacts check ratio
                if  (aspect_ratio>(1./ASPECT_RATIO) and aspect_ratio<ASPECT_RATIO) :
                    cv2.drawContours(mask2, [c], -1, 0, -1)
                    quality_meter=quality_meter+1
                    quality_mass= quality_mass+ cv2.contourArea(c)
            else:
                cv2.drawContours(mask2, [c], -1, 0, -1)
                quality_meter=quality_meter+1
                quality_mass= quality_mass+ cv2.contourArea(c)
    
    quality_percent = float(quality_meter)/ (len(cnt)+1)
    quality_mass_percent  =  float(quality_mass)/ (total_mass+1)


    mask2=1- mask2/mask2.max()  
    tophat= tophat*mask2 
    
    if silence==False:  
        titles = ['Refined Contour mask', 'Refined Tophat']
        images = [mask2, tophat]

        plt.figure() 
        
        for i in xrange(2):
            plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        
        plt.show()
       
        print ("Imaging quality (count)=%.2f (large is good)" %quality_percent)
        print ("Imaging quality (area) =%.2f (large is good)" %quality_mass_percent)
       
    return tophat, mask2
    
    
def DetectHE(img, silence=True):
    hist = cv2.calcHist([img],[0],None,[4],[0,256])
    print ("Hist[0]=%3.3f" %hist[0])

    if (hist[0]<500000):
        gamma= abs(550000-hist[0])/200000. +1
        img=GammaCorrection(img,gamma)
    else:
        gamma=1
    
    hist = cv2.calcHist([img],[0],None,[4],[0,256])
    print ("After Gamma=%2.2f Hist[0]=%3.3f" %(gamma,hist[0]) )  
    print ("channel mean=%3.3f" %np.mean(img))
    
    dilate, closing, opening = BasicMorphology(img, DIL=5, CLO=4, silence=silence)
            
    circular_mask, fill_mask, circular_inv, total_mask = CircularDetectMasking(img, opening, silence=silence)

    tophat, mask2 = FeaturesDetection(opening, total_mask, silence=silence) 
    
    return tophat, mask2
    

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

   





    
    
    
    
    
    
    
