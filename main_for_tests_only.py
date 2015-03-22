# -*- coding: utf-8 -*-
#auto path loading in order to run easily from terminal
import sys
sys.path.append('./ImageProcessing')
sys.path.append('./DataCreation')
import cv2

import ImageProcessing
import LoadData


names, labels = LoadData.ImageDatasetCreation(csv_name='./CSV/trainLabels.csv', labels_idx=[2,1], number_of_data=[30,30], LRB='both')

names_labels= (names, labels )

'''
for i in range(names_labels[1].shape[0]):
    if names_labels[1][i] == 4:
        names_labels[1][i] = 1
'''       
        
counter = 1
for i in range(1):#range(names.shape[0]):
     
    
    print 'Processing image %d'%counter    
    counter = counter + 1
    
    #name='95_right'
    #name='621_right' #not solved

    #name='456_left' 
    #name='363_right' 

    img_name = '../data/resized/%s.jpg'%names[i]
    img_name_temp = '../%s.jpg'%names[i]
    
    img = ImageProcessing.LoadImage(img_name)

    r,g,b = ImageProcessing.SplitImage(img, silence=True)
    


    features, mask2 = ImageProcessing.DetectHE(g, silence=False)
    
    cropped_image = ImageProcessing.CropImage(g, features, silence=True)
    plt.figure()
    plt.imshow(cropped_image, cmap = 'gray')
    plt.show()
    
    plt.figure()
    plt.imshow(g, cmap = 'gray')
    plt.show()
    
    print '--->Image Name:%s, Image label=%d '% (names[i],labels[i])
        
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
