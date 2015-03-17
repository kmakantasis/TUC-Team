# -*- coding: utf-8 -*-
#auto path loading in order to run easily from terminal
import sys
sys.path.append('./ImageProcessing')
sys.path.append('./DataCreation')
import cv2

import ImageProcessing
import LoadData


names, labels = LoadData.ImageDatasetCreation(csv_name='./CSV/trainLabels.csv', number_of_data=[500, 500])
for i in range(labels.shape[0]):
    if labels[i] == 4:
        labels[i] = 1
counter = 1
for name in names[:1]: 
    
    print 'Processing image %d'%counter    
    counter = counter + 1
    
    #img_name = '../data/resized/%s.jpg'%name
    
    img_name = '../data/resized/%s.jpg'%name
    

    
    img = ImageProcessing.LoadImage(img_name)

    r,g,b = ImageProcessing.SplitImage(img, silence=True)

    #features, mask2 = ImageProcessing.DetectHE(g, silence=False)
         
   # ImageProcessing.Rotation_Correct(r,g, name.split('_')[1], silent=True)
    #ImageProcessing.Rotation_Correct(r,g, name.split('_')[1], silent=False)
    
    #ret,thresh_g = cv2.threshold(g,150,250,cv2.THRESH_BINARY)  


    g = cv2.blur(g,(10,10))
    r = cv2.blur(r,(20,20))
    g, opening, closing=ImageProcessing.BasicMorphology(g, DIL=2, CLO=4, silence=True)
    dilate, opening, r=ImageProcessing.BasicMorphology(r, DIL=5, CLO=4, silence=True)
    
    r=ImageProcessing.GammaCorrection(r,2)
    g=ImageProcessing.GammaCorrection(g,1.5)
    #dilate, opening, r2=ImageProcessing.BasicMorphology(g, DIL=5, CLO=4, silence=True)
        
    #r2=ImageProcessing.GammaCorrection(r,3)
    #g2=g
    #ret,r = cv2.threshold(r,150,255,cv2.THRESH_BINARY)
   # cv2.equalizeHist( r, r );
    ImageProcessing.Rotation_Correct(r,g, name.split('_')[1], silent=False)

'''
    cropped_image = ImageProcessing.CropImage(g, features, silence=True)

    res = cv2.resize(cropped_image, (250, 250),  interpolation = cv2.INTER_AREA)

    out_name = '../data/input/%s.jpg'%name
    ret = cv2.imwrite(out_name, res)
'''
