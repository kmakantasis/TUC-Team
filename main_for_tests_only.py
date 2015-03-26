# -*- coding: utf-8 -*-
#auto path loading in order to run easily from terminal
import sys
import matplotlib.image as mpimg

sys.path.append('./ImageProcessing')
sys.path.append('./DataCreation')
import cv2
import numpy
import ImageProcessing
import ImageUtils
import LoadData


names, labels = LoadData.ImageDatasetCreation(csv_name='./CSV/trainLabels.csv', labels_idx=[3,4], number_of_data=[300,300], LRB='both')

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

    #name='456_left' 
    #name='1430_left'
    name='11031_right' #HE


    img_name = '../data/train_resized/%s.jpg'%name
    img_name_temp = '../%s.jpg'%name
    
    img = ImageUtils.LoadImage(img_name)

    r,g,b = ImageUtils.SplitImage(img, silence=True)
    
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b #cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray=255*(gray/gray.max())
   # gray= gray.flatten()
    
    gray=numpy.uint8(gray)
     

    features, mask2 = ImageProcessing.DetectHE(gray, gamma_offset=-0.6, silence=True)

    
    #ImageProcessing.DetectMicroAN(g)
    
    #cropped_image = ImageProcessing.CropImage(g, features, silence=True)
    
    #ImageProcessing.TriangularMasking()
    plt.figure()
    image = mpimg.imread(img_name)
    plt.imshow(image)
    plt.show()
    
    print '--->Image Name:%s, Image label=%d '% (name,label)
        
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
