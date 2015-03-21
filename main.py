# -*- coding: utf-8 -*-
#auto path loading in order to run easily from terminal
import sys
sys.path.append('./ImageProcessing')
sys.path.append('./DataCreation')
import cv2
import numpy as np
import ImageProcessing
import LoadData

names, labels = LoadData.ImageDatasetCreation(csv_name='./CSV/trainLabels.csv', labels_idx=[0,2,3,4], number_of_data=[1000, 300, 350, 350], LRB='left')

counter = 1
for name in names: 
    
    print name
    
    print 'Processing image %d'%counter 
    print ('Image Name: '),(name)
    print  'Image Label: %f'%labels[counter-1]
    
    counter = counter + 1
    
    img_name = 'data/resized/%s.jpg'%name
    img = ImageProcessing.LoadImage(img_name)

    r,g,b = ImageProcessing.SplitImage(img, silence=True)
    
    #----Antonis new stuff: Flip and Rotation Correct -------    
    g_flip_rotated, white_xy, dark_xy  = ImageProcessing.Flip_Rotation_Correct(r,g, name.split('_')[1], silence=True)    
    g=g_flip_rotated
    #----end Antonis new stuff ---

    #-->Kostas if you like to bulid the triangular mask use the white_xy and dark_xy cooridnate tuples 

    features, mask2 = ImageProcessing.DetectHE(g, silence=True)
    
    cropped_image = ImageProcessing.CropImage(g, features, silence=True)

    res = cv2.resize(cropped_image, (250, 250),  interpolation = cv2.INTER_AREA)

    out_name = 'data/input/%s.jpg'%name
#    ret = cv2.imwrite(out_name, res)

#names_input, labels_input = LoadData.InputDataset(csv_name='./CSV/trainLabels.csv', input_folder='data/input')
#
#labels_0_1 = np.zeros((2000,))
#for i in range(labels.shape[0]):
#    if labels_input[i][0] > 0:
#        labels_0_1[i] = 1
#        
#names_input = np.reshape(names_input, (2000, ))