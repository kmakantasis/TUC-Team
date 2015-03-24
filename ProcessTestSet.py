# -*- coding: utf-8 -*-
#auto path loading in order to run easily from terminal
import sys
sys.path.append('./ImageProcessing')
sys.path.append('./DataCreation')
import cv2
import numpy as np
import ImageProcessing
import LoadData
import multiprocessing
import os

def start_process():
    print 'Starting in multi processing', multiprocessing.current_process().name

def worker(name):
    print 'Worker'
   
  #  counter = 1
    #for name in names: 
        
    print name
    
    #print 'Processing image %d'%counter 
    print ('Processing Image Name: '),(name)
   # print  'Image Label: %f'%labels[counter-1]
    
   
    is_file=os.path.isfile('../data/input_test/%s.jpg'%name)
    if is_file==1:
        target_size= os.path.getsize('../data/input_test/%s.jpg'%name)
        if target_size>0:
           
            print 'Filename: %s exists, .....skippig: '%(name)
            return 0
        
    
   # counter = counter + 1
    
    img_name = '../data/test_resized/%s.jpg'%name
    print ('Complete Image path: '),(img_name)
    
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

    out_name = '../data/input_test/%s.jpg'%name
    ret = cv2.imwrite(out_name, res)
    return 1
 
#choose the mode to run
run_multiprocessor=0


files = [f.split('.')[0] for f in os.listdir('../data/test_resized')]

#names, labels = LoadData.ImageDatasetCreation(csv_name='./CSV/trainLabels.csv', 
#                labels_idx=[0,1,2,3,4], 
#                number_of_data=[25810, 2443, 5292, 873, 708], LRB='both')
names=np.asarray(files)
if run_multiprocessor:
    pool_size = multiprocessing.cpu_count()*2
    pool = multiprocessing.Pool(processes=pool_size,initializer=start_process,
                                    maxtasksperchild=2)
    pool_outputs = pool.map(worker, names)
    pool.close()
    pool.join()
else:
    for name in names:
        print 'Starting in single processing'
        worker(name)
    
 
