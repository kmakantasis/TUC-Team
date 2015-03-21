# -*- coding: utf-8 -*-
import numpy as np
import time
import sys
import os
import theano
import theano.tensor as T
import CNNLoadData
import ConvPoolLayer
import MultiLayerPerceptron
import LogisticLayer
import LoadData


def test_cnn(names, labels, learning_rate=0.05, L_reg=0.005, n_epochs=200, nkerns=[3, 6, 12, 24, 48], batch_size=50):
    
    # Load dataset
    datasets = CNNLoadData.LoadData(names, labels, ratio=0.80)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # Construct the model
    print '... building the model'

    index = T.lscalar()  
    x = T.matrix('x')  
    y = T.ivector('y')  

    rng = np.random.RandomState(1234)
    
    layer0_input = x.reshape((batch_size, 1, 250, 250))
    layer0 = ConvPoolLayer.ConvPoolLayer(rng, 
                                         layer0_input, 
                                         filter_shape=(nkerns[0], 1, 3, 3),
                                         image_shape=(batch_size, 1, 250, 250))
                                         
    layer1 = ConvPoolLayer.ConvPoolLayer(rng,
                                         layer0.output,
                                         filter_shape=(nkerns[1], nkerns[0], 3, 3),
                                         image_shape=(batch_size, nkerns[0], 124, 124))
                                         
    layer2 = ConvPoolLayer.ConvPoolLayer(rng,
                                         layer1.output,
                                         filter_shape=(nkerns[2], nkerns[1], 5, 5),
                                         image_shape=(batch_size, nkerns[1], 61, 61))
                                         
    layer3 = ConvPoolLayer.ConvPoolLayer(rng,
                                         layer2.output,
                                         filter_shape=(nkerns[3], nkerns[2], 5, 5),
                                         image_shape=(batch_size, nkerns[2], 28, 28))
                                         
    layer4 = ConvPoolLayer.ConvPoolLayer(rng,
                                         layer3.output,
                                         filter_shape=(nkerns[4], nkerns[3], 5, 5),
                                         image_shape=(batch_size, nkerns[3], 12, 12))
                                         
                                         
    layer5 = MultiLayerPerceptron.MLP(rng,
                                      layer4.output.flatten(2),
                                      nkerns[4] * 4 * 4,
                                      64,
                                      2)
                            
#    layer5 = MultiLayerPerceptron.HiddenLayer(rng, 
#                                      layer4.output.flatten(2),
#                                      nkerns[4] * 4 * 4, 
#                                      64, 
#                                      activation=T.tanh)
#                                      
#    layer6 = LogisticLayer.LogisticLayer(layer5.output, 64, 2)
    
    cost = (layer5.negative_log_likelihood(y) + L_reg * layer5.L1)
    
    # Function to train the model
    #params = layer6.params + layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
    params = layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
    gparams = T.grad(cost, params)
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(params, gparams)]
    train_model = theano.function(inputs=[index],
                                  outputs=[cost],
                                  updates=updates,
                                  givens={x:train_set_x[index * batch_size: (index+1) * batch_size],
                                          y:train_set_y[index * batch_size: (index+1) * batch_size]})
                                          
    # Functions to test and validate the model
    valid_model = theano.function(inputs=[index],
                                  outputs=[layer5.errors(y)],
                                  givens={x:valid_set_x[index * batch_size: (index+1) * batch_size],
                                          y:valid_set_y[index * batch_size: (index+1) * batch_size]})
                                          
    train_error = theano.function(inputs=[index],
                                  outputs=[layer5.errors(y)],
                                  givens={x:train_set_x[index * batch_size: (index+1) * batch_size],
                                          y:train_set_y[index * batch_size: (index+1) * batch_size]})
                                          
    test_model = theano.function(inputs=[index],
                                 outputs=[layer5.errors(y)],
                                 givens={x:test_set_x[index * batch_size: (index+1) * batch_size],
                                         y:test_set_y[index * batch_size: (index+1) * batch_size]})
                                         
    print '... training the model'
    patience = 10000  
    patience_increase = 2  
    improvement_threshold = 0.995  
    validation_frequency = min(n_train_batches, patience / 2)
                                  
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            
            train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:
                
                train_losses = [train_error(i) for i in xrange(n_train_batches)]
                this_train_loss = np.mean(train_losses)
                print('epoch %i, minibatch %i/%i, train set error %f %%' %  (epoch, 
                                                                              minibatch_index + 1, 
                                                                              n_train_batches,
                                                                              this_train_loss * 100.))
                                                                              

                validation_losses = [valid_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %  (epoch, 
                                                                              minibatch_index + 1, 
                                                                              n_train_batches,
                                                                              this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss:

                    if this_validation_loss < best_validation_loss *  improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    
    names_input, labels_input = LoadData.InputDataset(csv_name='../CSV/trainLabels.csv', input_folder='../data/input')

    labels_0_1 = np.zeros((2000,))
    for i in range(labels_input.shape[0]):
        if labels_input[i][0] > 0:
            labels_0_1[i] = 1
        
    names_input = np.reshape(names_input, (2000, ))
   
    test_cnn(names_input, labels_0_1)