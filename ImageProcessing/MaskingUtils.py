# -*- coding: utf-8 -*-
from scipy import ndimage
import matplotlib.pyplot as plt
import math
import numpy as np
import ImageUtils as ImU
import ImageProcessing
import ContourProcessing as CntP
import MaskingUtils as Msk

import cv2
 
def TotalMask(img, silence=True): 

    simple_mask_cirlualr=Msk.CircularMaskSimple(img)
    x,y= Msk.Disc_Detect(img,'WHITE')
    optic_disc_mask= Msk.DiscMask(img, x,y,80)
   
    total_mask= optic_disc_mask*simple_mask_cirlualr #*total_mask*
    total_mask= total_mask/total_mask.max()
    if silence == False: ImU.PrintImg(total_mask,'total_mask')
    
    return total_mask
     


     
 
 
def CircularMaskSimple(img):
    
    img_h, img_w = img.shape
    x= int(img_w/2)
    y= int(img_h/2)
 
    black = np.ones(img.shape)
    cv2.circle(black, (x,y), int(0.9*img_h/2), 0, -1)  # -1 to draw filled circles
    

    mask = 1 -black
    #ImU.PrintImg(mask,'simple circular mask')
    return mask

 
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
    open_for_mask = ImU.GammaCorrection(opening,3)#cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
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



def Find_Majority(k):

    k=np.asarray(k)
    hits=list()
    for n in range (1, len(k) ) :
        dk=abs(k[n]-k[n-1])
        if dk <20:
            hits.append(k[n-1])
            hits.append(k[n])
            
    if len(hits)>0:
        return int ( sum(hits)/len(hits) )
    else: 
        return int ( sum(k)/len(k) )
        
    
def Disc_Detect(img2,disc_type,silence=True):

    '''
    Last Maintenance: Antonis
    hits.append(k[n-1])
    Function definition
    +++++++++++++++++++
            
        .. py:function:: Disc_Detect(image_channel, disc_type)

            Detects circular discs, DARK or WHITE on input image. Utilizes majority vote on various scales
            
            :param uint image_channel:Input grayscale channe 
            :param string disc_type: 'DARK' or 'WHITE' to increase robustnes                        
            :rtype: Returns center_x of detected disc 
            :rtype: Returns center_y of detected disc 
    ''' 
  


    #loading templates
    if disc_type =='DARK':
        template = cv2.imread('./ImageProcessing/fovea_template.jpg',0)
        disc_size=170 #initial scale
 

    elif disc_type =='WHITE':
        template = cv2.imread('./ImageProcessing/disc_template.jpg',0)
        disc_size=230  #initial scale
 
    else:
        print("ERROR invalid disc type")
    
    scales=[0.85, 0.95 , 1, 1.11 ]
     
    templates=[template]
      
    methods = ['cv2.TM_CCOEFF_NORMED'] #'cv2.TM_CCOEFF_NORMED',
   # Different methods to choose from
   # , 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
   # methods = ['cv2.TM_SQDIFF_NORMED','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF']
    all_centers=list()
    for meth in methods:
        for sc in scales:
            for templ in templates:
                templ = cv2.resize(templ, ( int(sc*disc_size), int(sc*disc_size) ) )
                
                w, h = templ.shape[::-1]
                img = img2.copy()
                method = eval(meth)
            
                # Apply template Matching
                res = cv2.matchTemplate(img,templ,method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
                # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    top_left = min_loc
                else:
                    top_left = max_loc
                
                          
                
                bottom_right = (top_left[0] + w, top_left[1] + h)
                
                center_x= (bottom_right[0]+top_left[0])/2
                center_y= (bottom_right[1]+top_left[1])/2
                
                all_centers.append ([center_x, center_y])
                
                
                if silence==False:
                    cv2.rectangle(img,top_left, bottom_right, 0, 14)
                    cv2.circle(img,(int(center_x),int(center_y)),10,(255,255,255),-11) 
        
                    plt.subplot(121),plt.imshow(res,cmap = 'gray')
                    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
                    plt.subplot(122),plt.imshow(img,cmap = 'gray')
                    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])             
                    plt.show()
                    print("Confidence of detection=%f") %(max_val) 
                    print ("Disc x=%d , y=%d ")  %(center_x,center_y)
      
    x_majority = Find_Majority([t[0] for t in all_centers])
    y_majority = Find_Majority([t[1] for t in all_centers])
    
    if silence==False:
        print ("Majority Vote Disc x=%d , y=%d ")  %(x_majority,y_majority)
                
    return x_majority, y_majority 
    
    
def DiscMask(img,x,y,r):
    black = np.ones(img.shape)
    cv2.circle(black, (x,y), int(r+15), 0, -1)  # -1 to draw filled circles
    return black
    
def Global_LR_Gradient(g, silence=False):
    '''
    Last Maintenance: Antonis
    Function definition
    +++++++++++++++++++
        .. py:function:: 

            Detects in horizontal direction the global gradient in order to flip or not.
            It is far more robust than using the location of Dark and White Discs.
              
    
    '''   

    w, h = g.shape[::-1]
    black_white = cv2.imread('./ImageProcessing/black_white.jpg',0)
    black_white = cv2.resize(black_white, (w, int(1*h) ))
    
    white_black = cv2.imread('./ImageProcessing/white_black.jpg',0)
    white_black = cv2.resize(white_black, (w, int(1*h )))
    method = ['cv2.TM_CCOEFF_NORMED']
    grad_templates=[black_white, white_black]
    
    hist = cv2.calcHist([g],[0],None,[4],[0,256])
    
    if (hist[0]<500000):
        gamma= abs(550000-hist[0])/250000. +1
        img_g=ImU.GammaCorrection(g,gamma)
    else:
        img_g=g
        gamma=1
        
    belief=list()
    for templ in grad_templates:
                
        w, h = templ.shape[::-1]
        img = img_g.copy()
         
        # Apply template Matching
        res = cv2.matchTemplate(img,templ,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        belief.append(max_val)
        #print("Confidence  max_val=%f") %(max_val)
        #print("Gradient type=%s") %(templ) 

    
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
               
        
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        center_x= (bottom_right[0]+top_left[0])/2
        center_y= (bottom_right[1]+top_left[1])/2
        
               
        if silence==False:
            cv2.rectangle(img,top_left, bottom_right, 0, 14)
            cv2.circle(img,(int(center_x),int(center_y)),10,(255,255,255),-11) 

            plt.subplot(121),plt.imshow(res,cmap = 'gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(img,cmap = 'gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])             
            plt.show()   
            print("Gradient  max_val=%f") %(max_val) 
            print ("Gradient x=%d , y=%d ")  %(center_x,center_y)    
    
    WHITE_DISC_ON_RIGHT = belief[0]>belief[1]
    return WHITE_DISC_ON_RIGHT
   
    
    
def Flip_Rotation_Correct(r,g, LR_check, silence=False):
    
    '''
    Last Maintenance: Antonis
    Function definition
    +++++++++++++++++++
            
        .. py:function:: Rotation_Correct(red_channel, green_channel, silence)

            Apply  Flip and Rotation correction on input image.
            
            :param uint8 red_channel:  Need red for white disc detection
            :param uint8 green_channel:Need green for dark disck
            :param string LR_check: Need to know if image is left or right
            :param boolean silence: default is True. Set to False to print the result.
               
            :rtype: uint8 green_img
                - two dimensional uint green channel numpy array corresponding to rotation corrected image.
            :rtype: uint tuple white_disc_xy
                  - coordinates of both discs to link with next masking methods
            :rtype: uint tuple dark_disc_xy
              - coordinates of both discs to link with next masking methods
 
    '''   
    
    g_original=g # Keep original green untouched for rotate or return
    # Basic morphology correction, it could be simplified

    g = cv2.blur(g,(10,10))
    r = cv2.blur(r,(20,20))
    g=ImageProcessing.Dilate(g, DIL=3)
    dilate, opening, r=ImageProcessing.BasicMorphology(r, DIL=5, CLO=4, silence=True)
    
    r=ImU.GammaCorrection(r,.8)
    g=ImU.GammaCorrection(g,.8)    

     
    # Detect the two discs
    w, h = r.shape[::-1]
    x1, y1 = Disc_Detect(r,'WHITE',silence)
    x2, y2 = Disc_Detect(g,'DARK',silence)
    
    #---------------additional FLip check------
    WHITE_ON_THE_RIGHT = Global_LR_Gradient(dilate, silence)
    
    #-----------------------------------------
          
    # --------Check if image is mirrored -----
    # we consider normal images those have a notch 
    #  on the side of the image (square, triangle, or circle) 
    
    dx= x1-x2   
    dy= y1-y2  
    if silence==False:    
        if dx>0 and WHITE_ON_THE_RIGHT==1:
            print("----PASS Consistency in L/R check-----")
        elif dx<0 and WHITE_ON_THE_RIGHT==0:
            print("----PASS Consistency in L/R check-----")        
        elif dx>0 and WHITE_ON_THE_RIGHT==0:
            print("----FAILED Consistency  in L/R check-----")       
        elif dx<0 and WHITE_ON_THE_RIGHT==1:
            print("----FAILED Consistency  in L/R check-----")        
    
    if silence==False:
        print ("Initial dx=%d " %dx)
        
        plt.imshow(g,cmap = 'gray')
        plt.title('Smoothed Input image, before Flip Rotation correction'), plt.xticks([]), plt.yticks([])
        plt.show()   
        
    # Do the left/right checking    
    if LR_check=='right':
        if WHITE_ON_THE_RIGHT: #if WHITE disc on the right
            INV=0 #'not_inverted'
        else:
            INV=1 #'inverted'
            x1=w-x1
            x2=w-x1
                                
    if LR_check=='left':
        if WHITE_ON_THE_RIGHT: #if WHITE disc on the right
            INV=1 #'inverted'
            x1=w-x1
            x2=w-x1                          
        else:
            INV=0 #'not_inverted'
            
    if INV==1:
        g_original=cv2.flip(g_original, 1)      #Flip is done 
        if silence==False:   
            print("*****Flip detected and done*******") 
            
            plt.imshow(g_original,cmap = 'gray')
            plt.title('Green flipped image'), plt.xticks([]), plt.yticks([])
            plt.show()
    else:
        if silence==False:   
            print("******Flip NOT detected*******") 
        
    dx= (x1-x2)
    dy= (y1-y2)            
    #------ end Check if image is mirrored 
         
    rads = math.atan2(dy,dx)                                  
    degs = math.degrees(rads)

    # refining degrees to be in the first and fourt quadratiles 
    if degs<-120:
        degs =180+degs
        if silence==False:   
            print ("Over 90 degrees Angle=%2.4f ,normalizing )" %degs)      
        
    if degs>120:
        degs =-180+degs    
        if silence==False: 
            print ("Over 90 degrees Angle=%2.4f ,normalizing )" %degs)
                                
    if silence==False:   
        print ("Disc1 x=%d , y=%d ")  %(x1,y1)
        print ("Fovea-Optic Disc Angle=%2.4f )" %degs)
    
    # Choose if to rotate or not
    rows,cols = g.shape
    if abs(degs)<20:
        M = cv2.getRotationMatrix2D((cols/2,rows/2),degs,1) #cols/2,rows/2 defines the center of rotation, last argument is scale
        rot_g = cv2.warpAffine(g_original,M,(cols,rows)) # Rotation is done
        if silence==False:    
            plt.imshow(rot_g,cmap = 'gray')
            plt.title('Rotation correct'), plt.xticks([]), plt.yticks([])
            plt.show()      
    else:
        rot_g=g_original
        if silence==False:
            print("Large angle detected. Probably an error. Prefer not to rotate")

    if silence==False:       
        print ('--->Image L/R: '),(LR_check)
    
    white_disc_xy= (x1, y1)
    dark_disc_xy = (x2,y2)
    return rot_g, white_disc_xy ,  dark_disc_xy  #return green channel rotated, and coordinates

 
    
    
    
    
    
    
