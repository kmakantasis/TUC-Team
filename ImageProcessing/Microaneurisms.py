# -*- coding: utf-8 -*-
import sys
#sys.path.append('../ImageProcessing')
sys.path.append('../DataCreation')
import numpy as np
import cv2
import matplotlib.pylab as plt
import matplotlib.image as mpimg
from scipy.ndimage.morphology import binary_fill_holes as fill_holes
import scipy.spatial.distance as distance
import sklearn.svm as SVM
from sklearn.metrics import confusion_matrix
#import score
import time   
import LoadData 



def EnhanceContrast(g, r=3, op_kernel=15, silence=True):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(op_kernel,op_kernel))
    opening = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel)
    
    g_copy = np.asarray(np.copy(g), dtype=np.float)

    m_f = np.mean(opening)
        
    u_max = 245; u_min = 10; t_min = np.min(g); t_max = np.max(g)

    idx_gt_mf = np.where(g_copy > m_f)
    idx_lt_mf = np.where(g_copy <= m_f)

    g_copy[idx_gt_mf] = -0.5 * ((u_max-u_min) / (m_f-t_max)**r) * (g_copy[idx_gt_mf]-t_max)**r + u_max
    g_copy[idx_lt_mf] = 0.5 * ((u_max-u_min) / (m_f-t_min)**r) * (g_copy[idx_lt_mf]-t_min)**r + u_min 

    if silence == False:
        plt.subplot(1,2,1)
        plt.imshow(g, cmap='gray')
        plt.title('Original image')
        plt.subplot(1,2,2)
        plt.imshow(g_copy, cmap='gray')
        plt.title('Enhanced image')
        plt.show()
        
    return g_copy
    

def RemoveNoise(enhanced, kernel_sz=3, silence=True):
    
    enh_inv = np.max(enhanced) - enhanced
    
    blur = cv2.GaussianBlur(enh_inv,(kernel_sz,kernel_sz),sigmaX=1., sigmaY=1.)
    
    blur = np.max(blur) - blur
        
    if silence==False:
        titles= ['Original Image', 'Blur']
        images = [enhanced, blur]
            
        for i in range(2):
            plt.subplot(1,2,i+1)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])

        plt.show()
        
    return blur
    

def DetectAneurysms(b_3, thres, silence=True):
    b_3_int = np.asarray(b_3, dtype=np.uint8)
    b_3_int = cv2.bitwise_not(b_3_int)
    ret, th3 = cv2.threshold(b_3_int, thres, 255, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
    
    closing = cv2.bitwise_not(closing)
    closing = fill_holes(closing).astype(np.uint8)
    closing = closing*255
    edges = cv2.Canny(closing, 50, 100, apertureSize=3)
    
    
    des = np.copy(edges)
    contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contour:
        area = cv2.contourArea(cnt)
        if area > 30 and area < 1500:
            rect = cv2.minAreaRect(cnt)
            ratio = rect[1][0] / rect[1][1]
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            circ_area = 3.145 * radius**2
            if ratio > 0.33 and ratio < 3. and circ_area/area < 3. :
                cv2.drawContours(des,[cnt],0,255,-1)
            
    des = cv2.medianBlur(des,5)
    
    if silence==False:
        idx = np.where(des==255)
        detections = np.copy(b_3)
        detections[idx] = 255.
    
        plt.subplot(2,2,1)
        plt.imshow(b_3, cmap='gray')

        plt.subplot(2,2,2)
        plt.imshow(closing, cmap='gray')
    
        plt.subplot(2,2,3)
        plt.imshow(edges, cmap='gray')
        
        plt.subplot(2,2,4)
        plt.imshow(detections, cmap='gray')

        plt.show()
    
    return des/255


if __name__ == '__main__':
    
    names, labels = LoadData.ImageDatasetCreation(csv_name='../CSV/trainLabels.csv',labels_idx=[0,1,2,3,4], number_of_data=[2012, 2443, 2012, 873, 708], LRB='both')
    
#    names = np.array(['16_right'])
    start_time = time.time()
    
    dataset = []
    for i in range(1):
#        print i+1
    
        filename = '../../data/train_resized/%s.jpg'%names[i]
        
        print labels[np.where(names==names[i])]
    
        original = mpimg.imread(filename)
        img = cv2.imread(filename)
        g = img[:,:,1]
    
        enhanced = EnhanceContrast(g, r=3, silence=True)
        
        b_3 = RemoveNoise(enhanced, silence=True)
    
        thres = 150
        detections = np.zeros((g.shape[0], g.shape[1]), dtype=np.uint8)
        while thres < 250:
            des = DetectAneurysms(b_3, thres, silence=True)
            detections = np.logical_or(detections, des)
            thres = thres + 20
   
        print("--- %s seconds ---" % (time.time() - start_time))
    
        idx = np.where(detections==1)
        det_over = np.copy(b_3)
        det_over[idx] = 255.   
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(b_3, cmap='gray')

        plt.subplot(2,2,2)
        plt.imshow(detections, cmap='gray')
    
        plt.subplot(2,2,3)
        plt.imshow(det_over, cmap='gray')
    
        plt.subplot(2,2,4)
        plt.imshow(original)
        
        plt.show()    
    
    
#        test = np.zeros((detections.shape[0], detections.shape[1]), dtype=np.uint8)
#        contour,hier = cv2.findContours(detections.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
#    
#        total_area = 0
#        aneurysms_no = len(contour)
#        centroids = []
#        for cnt in contour:
#            cv2.drawContours(test,[cnt],0,255,-1)
#            area = cv2.contourArea(cnt)
#            M = cv2.moments(cnt)
#            cx = int(M['m10']/M['m00'])
#            cy = int(M['m01']/M['m00'])
#            centroids.append(np.array((cx, cy)))
#            total_area = total_area + area
#    
#        if len(centroids) > 1:
#            distance_avg = np.mean(distance.pdist(np.asarray(centroids)))
#        else:
#            distance_avg = 0.0
#           
#           
#           
#        label = labels[np.where(names==names[i])][0]
#    
#        entry = np.array((aneurysms_no, total_area, distance_avg, label))
#    
#        dataset.append(entry)
#        
#    data = np.asarray(dataset)
#    features = data[:,0:3]
#    targets = data[:,3]
#        
#    X_train = features[0:7000,:]
#    Y_train = targets[0:7000]
#
#    X_test = features[7000:,:]
#    Y_test = targets[7000:]
#
#    clf = SVM.SVC(class_weight={0: 1.2, 1:1.2, 2:1.0, 3:3., 4:3.4})
#    clf.fit(X_train, Y_train)
#
#    predictions = clf.predict(X_train)  
#    cm = confusion_matrix(Y_train, predictions)
#    rater_a = list(predictions.astype(np.int))
#    rater_b = list(Y_train.astype(np.int))
#    k = score.kappa(rater_a, rater_b)
#        
#    print 'kappa score:%f'%k

        
   
#        plt.figure()
#        plt.imshow(test, cmap='gray')
#        plt.show()
#        
#        print np.mean(distance.pdist(np.asarray(centroids)))
    