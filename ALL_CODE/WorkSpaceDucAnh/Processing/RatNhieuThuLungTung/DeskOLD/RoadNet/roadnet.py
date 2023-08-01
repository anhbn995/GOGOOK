import os
import sys
import numpy as np
import tensorflow as tf

from models import SideNet
from models import RoadSurfaceNet
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

class RoadNet(object):
    def __init__(self, input_shape=(128,128,1)):
        self.input_shape=input_shape
        self.centerline_net = SideNet(name='centerline', input_shape=input_shape)
        self.edge_net = SideNet(name='edge', input_shape=input_shape)
        self.surface_net = RoadSurfaceNet(input_shape=input_shape)
        self.beta = 0.1 ### For balanced crossentropy ###
        self.lambda_ = 2e-4 ### For generalization ###
        
        ### For weights of the loss components ###
        self.alpha = 1.0 # for balanced crossentropy
        self.gamma = 1.0 # for regularization 
        self.eta   = 1.0 # for generalization

    def weighted_binary_crossentropy(self):
        w1 = self.beta
        w2 = 1 - self.beta
    
        def loss(y_true, y_pred):
            # transform predicted map to a probability map 
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon()) # need epsilon to avoid absolute zeros
            ones = tf.ones_like(y_true) # create a mask
            msk = tf.equal(y_true, ones)

            ### Calculate weighted binary crossentropy loss with beta=0.1 to signify the imporance of loss where y==1 ###
            res, _ = tf.map_fn(lambda x: (tf.multiply(-tf.math.log(x[0]), w1) if x[1] is True else tf.multiply(-tf.math.log(1 - x[0]), w2), x[1]),
                               (y_pred, msk), dtype=(tf.float32, tf.bool))

            ### L2 normalization ###
            ### l2 norm = 1/(2|X|) * ||Y- P||2
            l2_norm = tf.nn.l2_normalize(y_pred - tf.cast(msk, dtype=tf.float32)) * (1/(y_pred.shape[1] * y_pred.shape[2]))
            

            return res + l2_norm

        return loss
    
    def get_model(self):
        inputs = Input(shape=self.input_shape)
        s1_surface, s2_surface, s3_surface, s4_surface, s5_surface, fused_surface = self.surface_net.get_model()(inputs)
        s1_edge, s2_edge, s3_edge, s4_edge, fused_edge = self.edge_net.get_model()([inputs, fused_surface])
        s1_line, s2_line, s3_line, s4_line, fused_line = self.centerline_net.get_model()([inputs, fused_surface])

        ### Basically naming layer for loss monitoring ###
        fused_surface = Lambda(lambda x: x, name='surface_final_output')(fused_surface)
        fused_edge    = Lambda(lambda x: x, name='edge_final_output')(fused_edge)
        fused_line    = Lambda(lambda x: x, name='line_final_output')(fused_line)
        
        s1_surface = Lambda(lambda x : x, name='surface_side_output_1')(s1_surface)
        s2_surface = Lambda(lambda x : x, name='surface_side_output_2')(s2_surface)
        s3_surface = Lambda(lambda x : x, name='surface_side_output_3')(s3_surface)
        s4_surface = Lambda(lambda x : x, name='surface_side_output_4')(s4_surface)
        s5_surface = Lambda(lambda x : x, name='surface_side_output_5')(s5_surface)

        s1_edge = Lambda(lambda x : x, name='edge_side_output_1')(s1_edge)        
        s2_edge = Lambda(lambda x : x, name='edge_side_output_2')(s2_edge)        
        s3_edge = Lambda(lambda x : x, name='edge_side_output_3')(s3_edge)        
        s4_edge = Lambda(lambda x : x, name='edge_side_output_4')(s4_edge) 

        s1_line = Lambda(lambda x : x, name='line_side_output_1')(s1_line)
        s2_line = Lambda(lambda x : x, name='line_side_output_2')(s2_line)
        s3_line = Lambda(lambda x : x, name='line_side_output_3')(s3_line)
        s4_line = Lambda(lambda x : x, name='line_side_output_4')(s4_line)
        
        model = Model(inputs=inputs, outputs=[fused_surface, fused_edge, fused_line, s1_surface, s2_surface, s3_surface, s4_surface,s5_surface, \
                s1_edge, s2_edge, s3_edge, s4_edge, \
                s1_line, s2_line, s3_line, s4_line])

        return model

