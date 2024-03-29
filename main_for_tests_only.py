# -*- coding: utf-8 -*-
#auto path loading in order to run easily from terminal
import sys
import matplotlib.pyplot as plt

sys.path.append('./ImageProcessing')
sys.path.append('./DataCreation')
import cv2
import numpy
import ImageProcessing as ImP
import RetinalSegmentation as RetSeg
import MaskingUtils as Msk
import Microaneurisms as MicroAns
 
import LoadData
import ImageUtils as ImU
import numpy as np


names, labels = LoadData.ImageDatasetCreation(csv_name='./CSV/trainLabels.csv', labels_idx=[0,1,2,3,4], number_of_data=[300,300,300,300,300])

names_labels= (names, labels )

'''
for i in range(names_labels[1].shape[0]):
    if names_labels[1][i] == 4:
        names_labels[1][i] = 1
'''       
        
counter = 1
for i in range(1):#range(names.shape[0]):
    name=names[i]
    label=labels[i]
    
    print 'Processing image %d'%counter    
    counter = counter + 1
    
    #name='229_left' #not solved
    #name='16_right'
    
    #name='818_left'
    
    #name='16114_right' 
    #name='1430_left'
    #name='11031_right' #HE done
    #name ='10321_left' #HE not done
    #name ='19116_right' #done

    #name ='2273_right'
    #name ='10904_right'
    #name ='11267_left'
    #name ='1639_left' #check

    #name ='5140_left' #no XE, more subtle deformations

    
    #name ='1008_right'
    #name ='4130_left'
    #name ='1099_right'

    img_name = '../data/train_resized/%s.jpg'%name
    
    #img_name = './blob.jpg'
    img_name_temp = '../%s.jpg'%name
    
    img = ImU.LoadImage(img_name)

    r,g,b = ImU.SplitImage(img, silence=True)
    
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b #cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray=255*(gray/gray.max())
   # gray= gray.flatten()
    
    gray=np.uint8(gray)
    
    
    #img2 =  cv2.imread(img_name)
   
    '''
    WHITE_DISC=ImU.ExtractPatch_W (g, 150)
    RetSeg.DetectFlow_1(WHITE_DISC, kernel_divide=15)
    
    DARK_DISC=ImU.ExtractPatch_B (g, 180) 
    RetSeg.DetectFlow_1(DARK_DISC,kernel_divide=8)
    '''
    
    
    #-------------------microans-------------------
    g2=ImU.HistAdjust(g)
    total_mask= Msk.CircularMaskSimple(g2)
    Vessels, theta_masked = RetSeg.DetectVessels(g,total_mask, kernel_divide=4, ContureFilter=True, silence=True)
    Vessels_mask=1-Vessels
    #Vessels_mask=ImP.Erode(Vessels_mask)
    ImU.PrintImg(Vessels_mask,'Vessels_mask')
    ImU.PrintImg(total_mask,'total_mask')
    g=g#*total_mask
    
    enhanced = MicroAns.EnhanceContrast(g, r=3, silence=True)
    
    b_3 = MicroAns.RemoveNoise(enhanced, silence=True)

    thres = 150
    detections = np.zeros((g.shape[0], g.shape[1]), dtype=np.uint8)
    while thres < 250:
        des = MicroAns.DetectAneurysms(b_3, thres, silence=True)
        detections = np.logical_or(detections, des)
        thres = thres + 20
   
   
    ImU.PrintImg(detections,'detections')
    detections=detections*Vessels_mask
    ImU.PrintImg(detections,'detections filtered')
    
    idx = np.where(detections==1)
    det_over = np.copy(b_3)
    det_over[idx] = 255.  
    
    ImU.PrintImg(det_over,'det_over')
 
    
    ''' 
    #---------HUE processing Example--------
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    ImU.AverageHue(hsv_img)
    d_theta= 360. / 8
    for theta in np.arange(0, 360,d_theta/2):
        ORANGE_MIN = np.array([theta, 50, 50],np.uint8)
        ORANGE_MAX = np.array([theta+d_theta, 250, 250],np.uint8)
        orange_mask = cv2.inRange(hsv_img, ORANGE_MIN, ORANGE_MAX)
        ImU.PrintImg(orange_mask,'orange_mask')
        print('d_theta=%d' % theta)
    '''
    '''
    #-------------kmeans example---------------
    labels,kmenas_out=ImU.kmeans(img)
    ImU.PrintImgColor(kmenas_out,'kmenas_out')
    '''
    
    #labels,g_kmenas_out=ImU.kmeans(g)
    #ImU.PrintImg(g_kmenas_out,'g_kmenas_out')
    
    #total_mask=Msk.TotalMask(g)
    #vessels_mask= RetSeg.DetectVessels(g,Msk.TotalMask(g), silence=False )

    #g2=ImU.BandCorrection(g,160,255, 0.6)
    # vessels_mask= ImageProcessing.DetectVessels(g2 )
  
    #ImU.PrintImg(img2,'img2')
    
    #vessels_mask= ImageProcessing.DetectVessels(img2) 
   
    #ImageProcessing.MatchedFilter(g)
  
    #circles=ImageProcessing.find_circles(g)
    
    #ret = cv2.imwrite(img_name_temp, vessels)
    #tophat=ImageProcessing.DetectHE(g, gamma_offset=0, silence=True)
    #ImageProcessing.MatchedFilter(g)
    
    #vessels_mask=ImageProcessing.MatchedFilter2(g)
    #ImU.PrintImg(vessels_mask,'vessels_mask')

    
    #features, mask2 = ImageProcessing.DetectHE(g, gamma_offset=-0.6, silence=True)
    #features, mask2 = ImageProcessing.DetectHE(g, gamma_offset=-0.6, silence=True)

    #ImageProcessing.DetectMicroAN(g)
    
    #cropped_image = ImageProcessing.CropImage(g, features, silence=True)
    
    #ImageProcessing.TriangularMasking()   
    
    plt.figure()
    image = plt.imread(img_name)
    plt.imshow(image)
    plt.show()
     
    print '--->Index=%d ,Image Name:%s, Image label=%d '% (i, name,label) 
        
'''
    g_rotated, white_xy, dark_xy  = ImageProcessing.Flip_Rotation_Correct(r,g, name.split('_')[1], silence=True)
    #ret = cv2.imwrite(img_name_temp, g_rotated)
    
    import matplotlib.pylab as plt
    
    plt.figure()
    plt.imshow(g, cmap = 'gray')
    plt.show()
    
    plt.figure()
    plt.imshow(g_rotated, cmap = 'gray')
    plt.show()
    '''
#    cropped_image = ImageProcessing.CropImage(g, features, silence=True)
#
#    res = cv2.resize(cropped_image, (250, 250),  interpolation = cv2.INTER_AREA)
#
#    out_name = '../data/input/%s.jpg'%name
#    ret = cv2.imwrite(out_name, res)
#
