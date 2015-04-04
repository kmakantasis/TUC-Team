# -*- coding: utf-8 -*-
#auto path loading in order to run easily from terminal
import sys
import matplotlib.pyplot as plt

sys.path.append('./ImageProcessing')
sys.path.append('./DataCreation')
import cv2
import numpy
import ImageProcessing
import RetinalSegmentation as RetSeg
 
import LoadData
import ImageUtils as ImU
import numpy as np


names, labels = LoadData.ImageDatasetCreation(csv_name='./CSV/trainLabels.csv', labels_idx=[2,3,4], number_of_data=[300,300,300])

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

    # name ='2273_right'
    #name ='10904_right'
    #name ='11267_left'
    #name ='1639_left' #check

    #name ='5140_left' #no XE, more subtle deformations

    #name ='5032_left'

    img_name = '../data/train_resized/%s.jpg'%name
    
    #img_name = './blob.jpg'
    img_name_temp = '../%s.jpg'%name
    
    img = ImU.LoadImage(img_name)

    r,g,b = ImU.SplitImage(img, silence=True)
    
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b #cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray=255*(gray/gray.max())
   # gray= gray.flatten()
    
    gray=np.uint8(gray)
  
    vessels_mask= RetSeg.DetectVessels(g )

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
   # ImU.PrintImg(vessels_mask,'vessels_mask')

    
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
