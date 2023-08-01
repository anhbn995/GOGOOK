import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

class RoadSurfaceNet(object):
    def __init__(self, input_shape=(128,128,3), name='surface'):
        self.input_shape=input_shape
        self.name=name
        
    def selu(self, x):
        alpha = 1.67326324
        scale = 1.05070098

        mask_1 = tf.cast((x > 0), dtype=tf.float32)
        mask_2 = tf.cast((x < 0), dtype=tf.float32)

        selu_ = (x * scale * mask_1) + ((scale * alpha * (tf.math.exp(x) - 1)) * mask_2)

        return selu_

    def get_model(self):
        ### Stage 1 ###
        inputs = Input(shape=self.input_shape)
        conv_1 = Conv2D(64, kernel_size=(3,3), activation='selu', padding='same')(inputs)
        conv_1 = Conv2D(64, kernel_size=(3,3), activation='selu', padding='same')(conv_1)
        pool_1 = MaxPooling2D(pool_size=(2,2))(conv_1)

        ### First output ###
        ### Size = W * H * 1 ###
        side_1 = Conv2D(1, kernel_size=(1,1), activation='relu', padding='same', name='surface_output_1', kernel_regularizer=l2(l2=1e-4))(conv_1)

        ### Stage 2 ###
        conv_2 = Conv2D(128, kernel_size=(3,3), activation='selu', padding='same')(pool_1)
        conv_2 = Conv2D(128, kernel_size=(3,3), activation='selu', padding='same')(conv_2)
        pool_2 = MaxPooling2D(pool_size=(2,2))(conv_2)

        ### Second output ###
        ### Size = W/2 * H/2 * 1 ###
        side_2 = Conv2D(1, kernel_size=(1,1), activation='relu')(conv_2)
        side_2 = Conv2DTranspose(1, kernel_size=(2,2), strides=(2,2), padding='same', name='surface_output_2', kernel_regularizer=l2(l2=1e-4))(side_2)

        ### Stage 3 ###
        conv_3 = Conv2D(256, kernel_size=(3,3), activation='selu', padding='same')(pool_2)
        conv_3 = Conv2D(256, kernel_size=(3,3), activation='selu', padding='same')(conv_3)
        conv_3 = Conv2D(256, kernel_size=(3,3), activation='selu', padding='same')(conv_3)
        pool_3 = MaxPooling2D(pool_size=(2,2))(conv_3)

        ### Third output ###
        ### Size = W/4 * H/4 * 1 ###
        side_3 = Conv2D(1, kernel_size=(1,1), activation='relu')(conv_3)
        side_3 = Conv2DTranspose(1, kernel_size=(4,4), strides=(4,4), padding='same', name='surface_output_3', kernel_regularizer=l2(l2=1e-4))(side_3)

        ### Stage 4 ###
        conv_4 = Conv2D(512, kernel_size=(3,3), activation='selu', padding='same')(pool_3)
        conv_4 = Conv2D(512, kernel_size=(3,3), activation='selu', padding='same')(conv_4)
        conv_4 = Conv2D(512, kernel_size=(3,3), activation='selu', padding='same')(conv_4)
        pool_4 = MaxPooling2D(pool_size=(2,2))(conv_4)

        ### Fourth output ###
        ### Size = W/8 * H/8 * 1 ###
        side_4 = Conv2D(1, kernel_size=(1,1), activation='relu')(conv_4)
        side_4 = Conv2DTranspose(1, kernel_size=(8,8), strides=(8,8), padding='same', name='surface_output_4', kernel_regularizer=l2(l2=1e-4))(side_4)

        ### Stage 5 ###
        conv_5 = Conv2D(512, kernel_size=(3,3), activation='selu', padding='same')(pool_4) 
        conv_5 = Conv2D(512, kernel_size=(3,3), activation='selu', padding='same')(conv_5)
        conv_5 = Conv2D(512, kernel_size=(3,3), activation='selu', padding='same')(conv_5)

        ### Fifth output ###
        ### Size = W/16 * H/16 * 1 ###
        side_5 = Conv2D(1, kernel_size=(1,1), activation='relu')(conv_5)
        side_5 = Conv2DTranspose(1, kernel_size=(16,16), strides=(16,16), padding='same', name='surface_output_5', kernel_regularizer=l2(l2=1e-4))(side_5)

        ### concatenated output ###
        concat = tf.keras.layers.concatenate((side_1, side_2, side_3, side_4, side_5), axis=3)
        concat = Conv2D(1, kernel_size=(1,1), activation='relu', name='surface_concat', kernel_regularizer=l2(l2=1e-4))(concat)


        ### We need to pass the original inputs and the final concat to centerline and edge networks ###
        model = Model(inputs=inputs, outputs=[side_1,side_2,side_3,side_4,side_5,concat],name=self.name)

        return model

class SideNet(object):
    def __init__(self, input_shape=(128,128,1), name='centerline_net'):
        self.input_shape=input_shape
        self.name = name

    def selu(self, x):
        alpha = 1.67326324
        scale = 1.05070098

        mask_1 = tf.cast((x > 0), dtype=tf.float32)
        mask_2 = tf.cast((x < 0), dtype=tf.float32)

        selu_ = (x * scale * mask_1) + ((scale * alpha * (tf.math.exp(x) - 1)) * mask_2)

        return selu_


    def get_model(self):
        ### Original Image ###
        input_1 = Input(shape=self.input_shape)
        ### The segmented map ###
        input_2 = Input(shape=self.input_shape)
        
        ### Concatenate the two inputs along the channels ###
        concat_input = tf.keras.layers.concatenate((input_1, input_2), axis=3)
        
        conv_1 = Conv2D(32, kernel_size=(3,3), activation='selu', padding='same')(concat_input)
        conv_1 = Conv2D(32, kernel_size=(3,3), activation='selu', padding='same')(conv_1)
        pool_1 = MaxPooling2D(pool_size=(2,2))(conv_1)

        ### First output ###
        side_1 = Conv2D(1, kernel_size=(1,1), activation='relu', name=self.name+'_output_1', kernel_regularizer=l2(l2=1e-4))(conv_1)

        conv_2 = Conv2D(64, kernel_size=(3,3), activation='selu', padding='same')(pool_1)
        conv_2 = Conv2D(64, kernel_size=(3,3), activation='selu', padding='same')(conv_2)
        pool_2 = MaxPooling2D(pool_size=(2,2))(conv_2)

        ### Second output ###
        side_2 = Conv2D(1, kernel_size=(1,1), activation='relu')(conv_2)
        side_2 = Conv2DTranspose(1, kernel_size=(2,2), strides=(2,2), padding='same', name=self.name+'_output_2', kernel_regularizer=l2(l2=1e-4))(side_2)

        conv_3 = Conv2D(128, kernel_size=(3,3), activation='selu', padding='same')(pool_2)
        conv_3 = Conv2D(128, kernel_size=(3,3), activation='selu', padding='same')(conv_3)
        pool_3 = MaxPooling2D(pool_size=(2,2))(conv_3)

        ### Third output ###
        side_3 = Conv2D(1, kernel_size=(1,1), activation='relu')(conv_3)
        side_3 = Conv2DTranspose(1, kernel_size=(4,4), strides=(4,4), padding='same', name=self.name+'_output_3', kernel_regularizer=l2(l2=1e-4))(side_3)

        conv_4 = Conv2D(256, kernel_size=(3,3), activation='selu', padding='same')(pool_3)
        conv_4 = Conv2D(256, kernel_size=(3,3), activation='selu', padding='same')(conv_4)

        ### Fourth output ###
        side_4 = Conv2D(1, kernel_size=(1,1), activation='relu')(conv_4)
        side_4 = Conv2DTranspose(1, kernel_size=(8,8), strides=(8,8), padding='same', name=self.name+'_output_4', kernel_regularizer=l2(l2=1e-4))(side_4)
        
        concat = tf.keras.layers.concatenate((side_1, side_2, side_3, side_4), axis=3)
        concat = Conv2D(1, kernel_size=(1,1), activation='relu', name=self.name+'_concat', kernel_regularizer=l2(l2=1e-4))(concat) 

        model = Model(inputs=[input_1, input_2], outputs=[side_1, side_2, side_3, side_4, concat], name=self.name)

        return model
