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




    
    
    
    
    
    
    
