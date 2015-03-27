# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import math
import cv2
 
 
def CNTCentroid(cnt):
    M = cv2.moments(cnt)
    if M['m00'] >0:
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])
    else: centroid_x, centroid_y = 0,0
            
        
    return centroid_x, centroid_y

def CNTRule_Sphericity(cnt,accept_ratio=0.7):
    #TODO
    center, radius= cv2.minEnclosingCircle(cnt)
    
    min_circle_area= 3.14159*radius**2
    cnt_area= cv2.contourArea(cnt)
    
    cnt_area_ratio = cnt_area/min_circle_area
    
    return cnt_area_ratio<accept_ratio
 
def CNTRule_KickOutCircular(img,cnt):
    img_h, img_w = img.shape
    centroid_x, centroid_y=CNTCentroid(cnt)
    
    eye_center_x= int(img_w/2)
    eye_center_y= int(img_h/2)
    eye_r= int(img_h/2)-0.08*img_h
    
    distance_from_center= math.sqrt( (centroid_x-eye_center_x)**2 + (centroid_y-eye_center_y)**2)
    
    return distance_from_center< eye_r
    
def CNTRule_Area(cnt, MIN_THRESHOLD, MAX_THRESHOLD):
    return MIN_THRESHOLD< cv2.contourArea(cnt) and cv2.contourArea(cnt) < MAX_THRESHOLD
    
def CNTRule_AspectRatio(c):
    ASPECT_RATIO=4
    x,y,w,h = cv2.boundingRect(c)
    aspect_ratio = float(w)/h
    
    return (aspect_ratio>(1./ASPECT_RATIO) and aspect_ratio<ASPECT_RATIO)  


def ContourFiltering(binary, silence=False):
  
    cnt = cv2.findContours(binary,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    

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
        
        mask2=1- mask2/mask2.max()
        
    '''     
    if silence==False:  
        titles = ['Refined Contour mask', 'Refined Tophat']
        images = [mask2, tophat]

        plt.figure() 
        
        for i in xrange(2):
            plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        
        plt.show()
      '''    
        
    return mask2                         
            
    #quality_percent = float(quality_meter)/ (len(cnt)+1)
   # quality_mass_percent  =  float(quality_mass)/ (total_mass+1)

    
    

    
    
    
    
