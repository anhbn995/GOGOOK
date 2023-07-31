#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 4 20:25:42 2021

@author: ducanh
"""
from keras.layers import (
    Input, Convolution2D, MaxPooling2D, UpSampling2D,
    Reshape, core, Dropout, Flatten,
    Activation, BatchNormalization, Lambda, Dense, Conv2D, Conv2DTranspose, concatenate,Permute,Cropping2D,Add)
from keras.models import Model
from keras.optimizers import Adam, Nadam, Adadelta,SGD
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.layers import concatenate as merge_l
import tensorflow as tf


def jaccard_coef(y_true, y_pred):
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def jaccard_coef_int(y_true, y_pred):
    smooth = 1e-12
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

def dice_loss3(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss3(y_true, y_pred)


def model_unet_zhixuhao(num_channel,size, num_class=1):
    inputs = Input((size, size,int(num_channel)))
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(drop5)
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)
    optimizer=Adam(lr=1e-5)
    model.compile(optimizer = optimizer, loss = bce_dice_loss, metrics = ['accuracy',jaccard_coef, jaccard_coef_int])
    
    return model


def unet_basic(num_channel,size):
    conv_params = dict(activation='relu', border_mode='same')
    merge_params = dict(axis=-1)
    inputs1 = Input((size, size,int(num_channel)))
    # inputs2 = Input((size, size,int(num_channel)))
    # merge_input = concatenate([inputs1, inputs2])
    conv1 = Convolution2D(32, (3,3), **conv_params)(inputs1)
    conv1 = Convolution2D(32, (3,3), **conv_params)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, (3,3), **conv_params)(pool1)
    conv2 = Convolution2D(64, (3,3), **conv_params)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, (3,3), **conv_params)(pool2)
    conv3 = Convolution2D(128, (3,3), **conv_params)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, (3,3), **conv_params)(pool3)
    conv4 = Convolution2D(256, (3,3), **conv_params)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, (3,3), **conv_params)(pool4)
    conv5 = Convolution2D(512, (3,3), **conv_params)(conv5)

    up6 = merge_l([UpSampling2D(size=(2, 2))(conv5), conv4], **merge_params)
    conv6 = Convolution2D(256, (3,3), **conv_params)(up6)
    conv6 = Convolution2D(256, (3,3), **conv_params)(conv6)

    up7 = merge_l([UpSampling2D(size=(2, 2))(conv6), conv3], **merge_params)
    conv7 = Convolution2D(128, (3,3), **conv_params)(up7)
    conv7 = Convolution2D(128, (3,3), **conv_params)(conv7)

    up8 = merge_l([UpSampling2D(size=(2, 2))(conv7), conv2], **merge_params)
    conv8 = Convolution2D(64, (3,3), **conv_params)(up8)
    conv8 = Convolution2D(64, (3,3), **conv_params)(conv8)

    up9 = merge_l([UpSampling2D(size=(2, 2))(conv8), conv1], **merge_params)
    conv9 = Convolution2D(32, (3,3), **conv_params)(up9)
    conv9 = Convolution2D(32, (3,3), **conv_params)(conv9)

    conv10 = Convolution2D(1, (1, 1), activation='sigmoid')(conv9)
    optimizer=Adam(lr=1e-3)
    # optimizer=SGD(lr=1e-3, decay=1e-8, momentum=0.9, nesterov=True)
    model = Model(input=inputs1, output=conv10)
    # model.compile(optimizer=optimizer,
    #             loss=binary_crossentropy,
    #             metrics=['accuracy', jaccard_coef, jaccard_coef_int])
    return model




from tensorflow.compat.v1.keras.backend import set_session
import tensorflow as tf
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, CSVLogger, History, EarlyStopping, LambdaCallback,ReduceLROnPlateau)
from tensorflow.keras import layers, backend, Model, utils

losses= tf.keras.losses.BinaryCrossentropy()
smooth = 1.

def BatchActivate(x):
    x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    x = layers.LeakyReLU()(x)
    return x

def convolution_block(x, n_filters, size, strides=(1,1), padding='same', activation=True):
    x = layers.Conv2D(n_filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, n_filters=16, batch_activate = False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, n_filters, (3,3))
    x = convolution_block(x, n_filters, (3,3))
    x = layers.Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x

def build_model(input_shape, n_filters, DropoutRatio=0.3):
    input_layer = tf.keras.Input(shape=input_shape)
    # 101 -> 50
    conv1 = layers.Conv2D(n_filters * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1, n_filters * 1)
    conv1 = residual_block(conv1, n_filters * 1, True)
    pool1 = layers.MaxPool2D((2, 2))(conv1)
    pool1 = layers.Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = layers.Conv2D(n_filters * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2, n_filters * 2)
    conv2 = residual_block(conv2, n_filters * 2, True)
    pool2 = layers.MaxPool2D((2, 2))(conv2)
    pool2 = layers.Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = layers.Conv2D(n_filters * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3, n_filters * 4)
    conv3 = residual_block(conv3, n_filters * 4, True)
    pool3 = layers.MaxPool2D((2, 2))(conv3)
    pool3 = layers.Dropout(DropoutRatio)(pool3)

    # Middle
    convm = layers.Conv2D(n_filters * 8, (3, 3), activation=None, padding="same")(pool3)
    convm = residual_block(convm, n_filters * 8)
    convm = residual_block(convm, n_filters * 8, True)

    # 12 -> 25
    deconv3 = layers.Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(DropoutRatio)(uconv3)

    uconv3 = layers.Conv2D(n_filters * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, n_filters * 4)
    uconv3 = residual_block(uconv3, n_filters * 4, True)

    # 25 -> 50
    deconv2 = layers.Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])

    uconv2 = layers.Dropout(DropoutRatio)(uconv2)
    uconv2 = layers.Conv2D(n_filters * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, n_filters * 2)
    uconv2 = residual_block(uconv2, n_filters * 2, True)

    # 50 -> 101
    deconv1 = layers.Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])

    uconv1 = layers.Dropout(DropoutRatio)(uconv1)
    uconv1 = layers.Conv2D(n_filters * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, n_filters * 1)
    uconv1 = residual_block(uconv1, n_filters * 1, True)

    # uconv1 = Dropout(DropoutRatio/2)(uconv1)
    output_layer_noActi = layers.Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = layers.Activation('sigmoid')(output_layer_noActi)
    
    model = Model(inputs=[input_layer], outputs=[output_layer])
    # optimizer=tf.keras.optimizers.Adam(learning_rate = 3e-04, clipvalue = 0.5)
    optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-05, momentum = 0.9)

    # from tensorflow.keras import metrics
    # model.compile(optimizer=optimizer,
    #             loss=losses,
    #             # metrics=['accuracy', f1_m, precision_m, recall_m]
    #               metrics=['accuracy', f1_m, precision_m, recall_m, dice_coef]
    # )
    return model

def recall_m(y_true, y_pred):
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + backend.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + backend.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+backend.epsilon()))

def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def cross_entropy_balanced(y_true, y_pred):
    _epsilon = _to_tensor(tf.keras.backend.epsilon(), y_pred.dtype.base_dtype)
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.math.log(y_pred/ (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)
    beta = count_neg / (count_neg + count_pos)
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)
    cost = tf.reduce_mean(cost * (1 - beta))

    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)

def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f * y_true_f) + tf.keras.backend.sum(y_pred_f * y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred) #+ cross_entropy_balanced(y_true, y_pred) # + losses(y_true, y_pred)

def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    inse = tf.math.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.math.reduce_sum(output * output, axis=axis)
        r = tf.math.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.math.reduce_sum(output, axis=axis)
        r = tf.math.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice

def binary_focal_loss_fixed(y_true, y_pred):
    gamma=2.
    alpha=.25
    y_true = tf.cast(y_true, tf.float32)
    epsilon = backend.epsilon()
    y_pred = backend.clip(y_pred, epsilon, 1.0 - epsilon)

    p_t = tf.where(backend.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_factor = backend.ones_like(y_true) * alpha

    alpha_t = tf.where(backend.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    cross_entropy = -backend.log(p_t)
    weight = alpha_t * backend.pow((1 - p_t), gamma)
    loss = weight * cross_entropy
    loss = backend.mean(backend.sum(loss, axis=1))
    return loss


"""
# Model planet
# input_size = 512
# num_band = 8
# FN_MODEL="/home/skm/SKM16/ALL_MODEL/Semantic_segmentation/PLANET_UINT8_MOSAIC_Water_Basic_512_8band_V1_rm_nodata/20220725_110928/20220725_110928_val_weights_last.h5"
# FN_MODEL="/home/skm/SKM16/ALL_MODEL/Semantic_segmentation/PLANET_UINT8_MOSAIC_Plant_Basic_512_8band_V0/20220726_095513/20220726_095513_val_weights_last.h5"
# FN_MODEL="/home/skm/SKM16/ALL_MODEL/Semantic_segmentation/PLANET_UINT8_MOSAIC_Forest_Basic_512_8band_V0/20220726_093057/20220726_093057_val_weights_last.h5"
# input_size = 256
# num_band = 3
# FN_MODEL="/home/skm/SKM/WORK/ALL_CODE/Model/20220828_175708_val_weights_last.h5"
# cnn_model = unet_basic(num_band, size=input_size)



# # # Luon Giu nguyen
# out_model_save_summary = FN_MODEL.replace('.h5', '_model_save_sumarry.h5')
# cnn_model.load_weights(FN_MODEL)
# cnn_model.save(out_model_save_summary)


# a = tf.keras.models.load_model(out_model_save_summary)
# a.summary()
"""



FN_MODEL="/home/skm/public_mount/DucAnhtmp/cloud/weight/cloud_only.h5"
cnn_model = build_model((None,None,4), 42)


# # Luon Giu nguyen
out_model_save_summary = FN_MODEL.replace('.h5', '_model_save_sumarry.h5')
cnn_model.load_weights(FN_MODEL)
cnn_model.save(out_model_save_summary)


a = tf.keras.models.load_model(out_model_save_summary)
a.summary()