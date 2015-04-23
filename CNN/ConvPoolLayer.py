# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample


class ConvPoolLayer(object):
    """ Convolution and pooling layer of a Convolutional Neural Network."""
    
    def __init__(self, rng, input, filter_shape, image_shape, pool_size=(2, 2), W=None, b=None, activation=T.tanh):

        # image_shape[1] and filter_shape[1] correspond 
        # to number of feature maps and must be equal        
        assert image_shape[1] == filter_shape[1]
        
        self.input = input
        
        # weights and bias initialization 
        fan_in = np.prod(filter_shape[1:])
        fan_out = ( filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(pool_size) )
        W_bound = np.sqrt( 6. / (fan_in + fan_out) )
        
        if W==None:
            self.W = theano.shared(value=np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                                    dtype=theano.config.floatX),
                                   name='W',
                                   borrow=True)
        else:
            self.W = theano.shared(value=W, name='W')
            
        if b==None:
            self.b = theano.shared(value=np.zeros( (filter_shape[0],), dtype=theano.config.floatX ), 
                                   name='b', 
                                   borrow=True )
        else:
            self.b= theano.shared(value=b, name='b')
                          
        # convolution of input feature maps with filters
        conv_out = T.nnet.conv2d(input=input, filters=self.W, image_shape=image_shape, filter_shape=filter_shape)
        
        # maxpooling to downsample each one of the feature maps
        pooled_out = downsample.max_pool_2d(input=conv_out, ds=pool_size, ignore_border=True)
        
        # feed downsampled feature maps to a non linear function
        self.output = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        
        self.params = [self.W, self.b]
        
        
#        def cnn_out(self, input):
#            return self.output