# -*- coding: utf-8 -*-

import ImageProcessing
import LoadData


names, labels = LoadData.ImageDatasetCreation(csv_name='data/trainLabels.csv')
 
img_name = 'data/resized/%s.jpg'%names[0]
img = ImageProcessing.LoadImage(img_name)

r,g,b = ImageProcessing.SplitImage(img, silence=True)

features, mask2 = ImageProcessing.DetectHE(g, silence=False)

cropped_image = ImageProcessing.CropImage(g, features, silence=False)