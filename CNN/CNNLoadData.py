# -*- coding: utf-8 -*-
import numpy as np
import cv2
import theano
import theano.tensor as T


def LoadData(names, labels, ratio=0.8):

    print '... loading data'

    dataset_x = np.zeros((names.shape[0], 250*250))
    dataset_y = np.zeros((names.shape[0]))

    for i in range(names.shape[0]):
        img_name = '../data/input/%s.jpg'%names[i]
        img = cv2.imread(img_name, 0)
        ret,thresh = cv2.threshold(img,1,1,cv2.THRESH_BINARY)
        img_flat = np.reshape(thresh, (1, -1))

        dataset_x[i,:] = img_flat
        dataset_y[i] = labels[i]

    train_end = int(ratio * names.shape[0])
    
    train_dataset_x = dataset_x[0:train_end, :]
    test_dataset_x = dataset_x[train_end:, :]
    valid_dataset_x = dataset_x[train_end:, :]

    train_dataset_y = dataset_y[0:train_end]
    test_dataset_y = dataset_y[train_end:]
    valid_dataset_y = dataset_y[train_end:]

    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        
        return shared_x, T.cast(shared_y, 'int32')


    test_set_x, test_set_y = shared_dataset(test_dataset_x, test_dataset_y)
    valid_set_x, valid_set_y = shared_dataset(valid_dataset_x, valid_dataset_y)
    train_set_x, train_set_y = shared_dataset(train_dataset_x, train_dataset_y)
    
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    
    return rval
    
    
def FeatureConstructionData(names, labels):

    print '... loading data'

    dataset_x = np.zeros((names.shape[0], 250*250))
    dataset_y = np.zeros((names.shape[0]))

    for i in range(names.shape[0]):
        img_name = '../data/input/%s.jpg'%names[i]
        img = cv2.imread(img_name, 0)
        ret,thresh = cv2.threshold(img,1,1,cv2.THRESH_BINARY)
        img_flat = np.reshape(thresh, (1, -1))

        dataset_x[i,:] = img_flat
        dataset_y[i] = labels[i]

    
    
    train_dataset_x = dataset_x
    train_dataset_y = dataset_y
    

    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        
        return shared_x, T.cast(shared_y, 'int32')

    train_set_x, train_set_y = shared_dataset(train_dataset_x, train_dataset_y)
    
    rval = [(train_set_x, train_set_y)]
    
    return rval