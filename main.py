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
    
    img_name = '../data/resized/%s.jpg'%name
    img = ImageProcessing.LoadImage(img_name)

    r,g,b = ImageProcessing.SplitImage(img, silence=True)

    features, mask2 = ImageProcessing.DetectHE(g, silence=False)

    cropped_image = ImageProcessing.CropImage(g, features, silence=True)

    res = cv2.resize(cropped_image, (250, 250),  interpolation = cv2.INTER_AREA)

    out_name = '../data/input/%s.jpg'%name
    ret = cv2.imwrite(out_name, res)

