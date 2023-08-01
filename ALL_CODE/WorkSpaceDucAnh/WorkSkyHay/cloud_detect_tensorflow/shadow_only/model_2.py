from tensorflow.compat.v1.keras.backend import set_session
import tensorflow as tf
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, CSVLogger, History, EarlyStopping, LambdaCallback,ReduceLROnPlateau)
from tensorflow.keras import layers, backend, Model, utils
import tensorflow.keras.activations as activations
# import tensorflow_addons.optimizers as optimizers
import tensorflow.keras.metrics as metrics
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
# import tensorflow_addons.losses as losses
import numpy as np
import cv2

smooth = 1.

def encoder_block(inputs, n_filters, kernel_size, strides):
    encoder = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(inputs)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation(activations.gelu)(encoder)
    encoder = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, padding='same', use_bias=False)(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation(activations.gelu)(encoder)
    return encoder


# Defining the decoder's up-sampling blocks.
def upscale_blocks(inputs):
    n_upscales = len(inputs)
    upscale_layers = []

    for i, inp in enumerate(inputs):
        p = n_upscales - i
        u = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2**p, padding='same')(inp)

        for i in range(2):
            u = layers.Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(u)
            u = layers.BatchNormalization()(u)
            u = layers.Activation(activations.gelu)(u)
            u = layers.Dropout(rate=0.4)(u)

        upscale_layers.append(u)
    return upscale_layers


# Defining the decoder's whole blocks.
def decoder_block(layers_to_upscale, inputs):
    upscaled_layers = upscale_blocks(layers_to_upscale)

    decoder_blocks = []

    for i, inp in enumerate(inputs):
        d = layers.Conv2D(filters=64, kernel_size=3, strides=2**i, padding='same', use_bias=False)(inp)
        d = layers.BatchNormalization()(d)
        d = layers.Activation(activations.gelu)(d)
        d = layers.Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(d)
        d = layers.BatchNormalization()(d)
        d = layers.Activation(activations.gelu)(d)

        decoder_blocks.append(d)

    decoder = layers.concatenate(upscaled_layers + decoder_blocks)
    decoder = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', use_bias=False)(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation(activations.gelu)(decoder)
    decoder = layers.Dropout(rate=0.4)(decoder)

    return decoder

def build_model(input_dim):
    inputs = layers.Input(input_dim)

    noisy_inputs = layers.GaussianNoise(stddev=0.2)(inputs)

    e1 = encoder_block(noisy_inputs, n_filters=32, kernel_size=3, strides=1)
    e2 = encoder_block(e1, n_filters=64, kernel_size=3, strides=2)
    e3 = encoder_block(e2, n_filters=128, kernel_size=3, strides=2)
    e4 = encoder_block(e3, n_filters=256, kernel_size=3, strides=2)
    e5 = encoder_block(e4, n_filters=512, kernel_size=3, strides=2)

    d4 = decoder_block(layers_to_upscale=[e5], inputs=[e4, e3, e2, e1])
    d3 = decoder_block(layers_to_upscale=[e5, d4], inputs=[e3, e2, e1])
    d2 = decoder_block(layers_to_upscale=[e5, d4, d3], inputs=[e2, e1])
    d1 = decoder_block(layers_to_upscale=[e5, d4, d3, d2], inputs=[e1])

    output = layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(d1) # tanh activation

    model = models.Model(inputs, output)
    losses= tf.keras.losses.BinaryCrossentropy()
    optimizer=tf.keras.optimizers.Adam(learning_rate = 3e-04, clipvalue = 0.5)
    # optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-05, momentum = 0.9)
    model.compile(
        optimizer=optimizer,#optimizers.Yogi(learning_rate=0.00025),
        loss=losses,#losses.sigmoid_focal_crossentropy,
        metrics=[metrics.MeanIoU(num_classes=2), 'accuracy', f1_m, precision_m, recall_m, dice_coef]
    )
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