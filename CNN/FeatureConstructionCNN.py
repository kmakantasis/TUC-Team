# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T
import CNNLoadData
import ConvPoolLayer
import LoadData
import TestSVM
import scipy.io as sio


def feature_construct(names, labels, nkerns=[3, 6, 12, 24, 48], batch_size=1):
    
    # Load dataset
    datasets = CNNLoadData.FeatureConstructionData(names, labels)

    train_set_x, train_set_y = datasets[0]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    
    d = sio.loadmat('weights.mat')
    
    layer0_w = d['layer0_w']
    layer0_b = np.reshape(d['layer0_b'], (-1,))
    layer1_w = d['layer1_w']
    layer1_b = np.reshape(d['layer1_b'], (-1,))
    layer2_w = d['layer2_w']
    layer2_b = np.reshape(d['layer2_b'], (-1,))
    layer3_w = d['layer3_w']
    layer3_b = np.reshape(d['layer3_b'], (-1,))
    layer4_w = d['layer4_w']
    layer4_b = np.reshape(d['layer4_b'], (-1,))


    # Construct the model
    print '... building the model'

    index = T.lscalar()  
    x = T.matrix('x')  
    #y = T.ivector('y')  

    rng = np.random.RandomState(1234)
    
    layer0_input = x.reshape((batch_size, 1, 250, 250))
    layer0 = ConvPoolLayer.ConvPoolLayer(rng, 
                                         layer0_input, 
                                         filter_shape=(nkerns[0], 1, 3, 3),
                                         image_shape=(batch_size, 1, 250, 250),
                                         W=layer0_w, 
                                         b=layer0_b)
                                         
    layer1 = ConvPoolLayer.ConvPoolLayer(rng,
                                         layer0.output,
                                         filter_shape=(nkerns[1], nkerns[0], 3, 3),
                                         image_shape=(batch_size, nkerns[0], 124, 124),
                                         W=layer1_w, 
                                         b=layer1_b)
                                         
    layer2 = ConvPoolLayer.ConvPoolLayer(rng,
                                         layer1.output,
                                         filter_shape=(nkerns[2], nkerns[1], 5, 5),
                                         image_shape=(batch_size, nkerns[1], 61, 61),
                                         W=layer2_w, 
                                         b=layer2_b)
                                         
    layer3 = ConvPoolLayer.ConvPoolLayer(rng,
                                         layer2.output,
                                         filter_shape=(nkerns[3], nkerns[2], 5, 5),
                                         image_shape=(batch_size, nkerns[2], 28, 28),
                                         W=layer3_w, 
                                         b=layer3_b)
                                         
    layer4 = ConvPoolLayer.ConvPoolLayer(rng,
                                         layer3.output,
                                         filter_shape=(nkerns[4], nkerns[3], 5, 5),
                                         image_shape=(batch_size, nkerns[3], 12, 12),
                                         W=layer4_w, 
                                         b=layer4_b)    


    feature_construction = theano.function(inputs=[index],
                                           outputs=[layer4.output],
                                           givens={x:train_set_x[index * batch_size: (index+1) * batch_size]})
                                         
 

    features = []
    print '... feature construction'
    for minibatch_index in xrange(n_train_batches):
        o = feature_construction(minibatch_index)
        output = np.reshape(o, (-1,))
        features.append(output)   
                          
    return features
    
    
if __name__ == '__main__':
    
    names_input, labels_input = LoadData.InputDataset(csv_name='../CSV/trainLabels.csv', input_folder='../data/input')

    labels_0_1 = np.zeros((2000,))
    for i in range(labels_input.shape[0]):
        if labels_input[i][0] > 0:
            labels_0_1[i] = 1
        
    names_input = np.reshape(names_input, (2000, ))
   
    features = feature_construct(names_input, labels_0_1)
    features = np.asarray(features)
    
    error, c_matrix = TestSVM.TestSVM(features, labels_0_1, silence=False)