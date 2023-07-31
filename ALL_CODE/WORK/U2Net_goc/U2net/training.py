from tensorflow.keras import layers, Model, regularizers, Input, backend
import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.compat.v1.keras.backend import set_session
import os, glob, shutil, warnings
import numpy as np
# from nebullvm import optimize_tf_model
from color_image import VisualEffect
from tqdm import tqdm

warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))


VisEff = VisualEffect()
class DataParser():
    def __init__(self, annotation):
        self.total_data = annotation
        self.batch_size = 2
        self.steps_per_epoch = int(len(self.total_data)//self.batch_size)
        self.check_batch = self.steps_per_epoch * self.batch_size
        self.augmentations = [self.flip_ud, self.flip_lr, self.rot90]
        self.num = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.num < self.check_batch:
            filename = self.total_data[self.num: self.num+self.batch_size]

            image, label = self.get_batch(filename)
            self.num += self.batch_size

            return image, label
        else:
            self.num = 0
            np.random.shuffle(self.total_data)
            raise StopIteration


    def get_batch(self, batch):
        images = []
        edgemaps = []
        for img_list in batch:
            im = np.load(img_list)[...,:3]
            em = np.load(img_list.replace('image', 'label'))
            im = VisEff(im)
            im = np.array(im/255.0, dtype=np.float32)
#             im = self.preprocess_image(im)
            em = em.astype(np.float32)

            for f in self.augmentations:
                if np.random.uniform()<0.20:
                    im, em=f(im, em)

            images.append(im)
            edgemaps.append(em)

        images   = np.asarray(images)
        edgemaps = np.asarray(edgemaps)

        return images, edgemaps
    
    def preprocess_image(self, image):
        image = image.astype(np.float32)
        image /= 255.
        mean= [0.31376, 0.35322, 0.25537]
        std= [0.12894, 0.09586, 0.08987]
        image -= mean
        image /= std
        return image
    
    def flip_ud(self, im, em):
        return np.flipud(im), np.flipud(em)

    def flip_lr(self, im, em):
        return np.fliplr(im), np.fliplr(em)

    def rot90(self, im, em):
        return np.rot90(im), np.rot90(em)
    
    def __len__(self):
        return self.steps_per_epoch
    

l2 = regularizers.l2
w_decay=1e-3
weight_init = tf.initializers.glorot_uniform()

def DoubleConvBlock(input, mid_features, out_features=None, stride=(1,1), use_bn=True,use_act=True):
    out_features = mid_features if out_features is None else out_features
    k_reg = None if w_decay is None else l2(w_decay)
    x = layers.Conv2D(filters=mid_features, kernel_size=(3, 3), strides=stride, padding='same', kernel_initializer=weight_init, kernel_regularizer=k_reg)(input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=out_features, kernel_size=(3, 3), strides=(1,1), padding='same', kernel_initializer=weight_init, kernel_regularizer=k_reg)(x)
    x = layers.BatchNormalization()(x)
    if use_act:
        x = layers.ReLU()(x)
    return x

def SingleConvBlock(input, out_features, k_size=(1,1),stride=(1,1), use_bs=False, use_act=False, w_init=None):
    k_reg = None if w_decay is None else l2(w_decay)
    x = layers.Conv2D(filters=out_features, kernel_size=k_size, strides=stride, padding='same',kernel_initializer=w_init, kernel_regularizer=k_reg)(input)
    if use_bs:
        x = layers.BatchNormalization()(x)
    if use_act:
        x = layers.ReLU()(x)
    return x

def UpConvBlock(input_data, up_scale):
    total_up_scale = 2 ** up_scale
    constant_features = 16
    k_reg = None if w_decay is None else l2(w_decay)
    features = []
    for i in range(up_scale):
        out_features = 1 if i == up_scale-1 else constant_features
        if i==up_scale-1:
            input_data = layers.Conv2D(filters=out_features, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu', kernel_initializer=tf.initializers.TruncatedNormal(mean=0.), kernel_regularizer=k_reg,use_bias=True)(input_data)
            input_data = layers.Conv2DTranspose(out_features, kernel_size=(total_up_scale,total_up_scale), strides=(2,2), padding='same', kernel_initializer=tf.initializers.TruncatedNormal(stddev=0.1), kernel_regularizer=k_reg,use_bias=True)(input_data)
        else:
            input_data = layers.Conv2D(filters=out_features, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu',kernel_initializer=weight_init, kernel_regularizer=k_reg,use_bias=True)(input_data)
            input_data = layers.Conv2DTranspose(out_features, kernel_size=(total_up_scale,total_up_scale),strides=(2,2), padding='same', use_bias=True, kernel_initializer=weight_init, kernel_regularizer=k_reg)(input_data)
    return input_data

def _DenseLayer(inputs, out_features):
    k_reg = None if w_decay is None else l2(w_decay)
    x, x2 = tuple(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=out_features, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer=weight_init, kernel_regularizer=k_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=out_features, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer=weight_init, kernel_regularizer=k_reg)(x)
    x = layers.BatchNormalization()(x)
    return 0.5 * (x + x2), x2

def _DenseBlock(input_da, num_layers, out_features):
    for i in range(num_layers):
        input_da = _DenseLayer(input_da, out_features)
    return input_da

def DexiNed(image_size, image_band_channel):
    img_input = Input(shape=(image_size,image_size,image_band_channel), name='input')

    block_1 = DoubleConvBlock(img_input, 32, 64, stride=(2,2),use_act=False)
    block_1_side = SingleConvBlock(block_1, 128, k_size=(1,1),stride=(2,2),use_bs=True, w_init=weight_init)

    # Block 2
    block_2 = DoubleConvBlock(block_1, 128, use_act=False)
    block_2_down = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(block_2)
    block_2_add = block_2_down + block_1_side
    block_2_side = SingleConvBlock(block_2_add, 256,k_size=(1,1),stride=(2,2),use_bs=True, w_init=weight_init)

    # Block 3
    block_3_pre_dense = SingleConvBlock(block_2_down,256,k_size=(1,1),stride=(1,1),use_bs=True,w_init=weight_init)
    block_3, _ = _DenseBlock([block_2_add, block_3_pre_dense], 2, 256)
    block_3_down = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(block_3)
    block_3_add = block_3_down + block_2_side
    block_3_side = SingleConvBlock(block_3_add, 512,k_size=(1,1),stride=(2,2),use_bs=True,w_init=weight_init)

    # Block 4
    block_4_pre_dense_256 = SingleConvBlock(block_2_down, 256,k_size=(1,1),stride=(2,2), w_init=weight_init)
    block_4_pre_dense = SingleConvBlock(block_4_pre_dense_256 + block_3_down, 512,k_size=(1,1),stride=(1,1),use_bs=True, w_init=weight_init)
    block_4, _ = _DenseBlock([block_3_add, block_4_pre_dense], 3, 512)
    block_4_down = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(block_4)
    block_4_add = block_4_down + block_3_side
    block_4_side = SingleConvBlock(block_4_add, 512,k_size=(1,1),stride=(1,1),use_bs=True, w_init=weight_init)

    # Block 5
    block_5_pre_dense_512 = SingleConvBlock(block_4_pre_dense_256, 512, k_size=(1,1),stride=(2,2), w_init=weight_init)
    block_5_pre_dense = SingleConvBlock(block_5_pre_dense_512 + block_4_down, 512,k_size=(1,1),stride=(1,1),use_bs=True, w_init=weight_init)
    block_5, _ = _DenseBlock([block_4_add, block_5_pre_dense], 3, 512)
    block_5_add = block_5 + block_4_side

    # Block 6
    block_6_pre_dense = SingleConvBlock(block_5, 256,k_size=(1,1),stride=(1,1),use_bs=True, w_init=weight_init)
    block_6, _ =  _DenseBlock([block_5_add, block_6_pre_dense], 3, 256)


    out_1 = UpConvBlock(block_1, 1)
    out_2 = UpConvBlock(block_2, 1)
    out_3 = UpConvBlock(block_3, 2)
    out_4 = UpConvBlock(block_4, 3)
    out_5 = UpConvBlock(block_5, 4)
    out_6 = UpConvBlock(block_6, 4)

    # concatenate multiscale outputs
    block_cat = tf.concat([out_1, out_2, out_3, out_4, out_5, out_6], 3)  # BxHxWX6
    block_cat = SingleConvBlock(block_cat, 1,k_size=(1,1),stride=(1,1), w_init=tf.constant_initializer(1/5))  # BxHxWX1
    
    block_cat = layers.Activation('sigmoid')(block_cat)
    out_1 = layers.Activation('sigmoid')(out_1)
    out_2 = layers.Activation('sigmoid')(out_2)
    out_3 = layers.Activation('sigmoid')(out_3)
    out_4 = layers.Activation('sigmoid')(out_4)
    out_5 = layers.Activation('sigmoid')(out_5)
    out_6 = layers.Activation('sigmoid')(out_6)

    model = Model(inputs=[img_input], outputs=[block_cat, out_1, out_2, out_3, out_4, out_5, out_6])
    # model.summary()

    return model

def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    # x = layers.BatchNormalization(axis=bn_axis, scale=False, center=False,name=bn_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x

def InceptionV3(img_input):
    output = []
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = layers.ZeroPadding2D(35)(img_input)
    x = conv2d_bn(x, 32, 3, 3, strides=(1, 1), padding='valid')
    x = conv2d_bn(x, 32, 3, 3)
    x = conv2d_bn(x, 64, 3, 3)
    output.append(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3)
    output.append(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1, padding='valid')

    branch5x5 = conv2d_bn(x, 48, 1, 1, padding='valid')
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, padding='valid')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1, padding='valid')
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1, padding='valid')

    branch5x5 = conv2d_bn(x, 48, 1, 1, padding='valid')
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, padding='valid')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, padding='valid')
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1, padding='valid')

    branch5x5 = conv2d_bn(x, 48, 1, 1, padding='valid')
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, padding='valid')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, padding='valid')
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed2')
    output.append(x)
    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding = 'valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, padding= 'valid')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding = 'valid')

    # branch_pool = layers.ZeroPadding2D(1)(x)
    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1, padding = 'valid')

    branch7x7 = conv2d_bn(x, 128, 1, 1, padding = 'valid')
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1, padding = 'valid')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, padding = 'valid')
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1, padding = 'valid')

        branch7x7 = conv2d_bn(x, 160, 1, 1, padding = 'valid')
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1, padding = 'valid')
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D( (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1, padding = 'valid')
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1, padding = 'valid')

    branch7x7 = conv2d_bn(x, 192, 1, 1, padding = 'valid')
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1, padding = 'valid')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, padding = 'valid')
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name='mixed7')
    output.append(x)

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1, padding = 'valid')
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1, padding = 'valid')
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1, padding = 'valid')

        branch3x3 = conv2d_bn(x, 384, 1, 1, padding = 'valid')
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1, padding = 'valid')
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1, padding = 'valid')
        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed' + str(9 + i))
    output.append(x)

    return output

def crop(d, region):
    x, y, h, w = region
    d1 = d[:, x:x + h, y:y + w, :]
    return d1

def DilConv(x, kernel_size, padding, dilation, stride = 1, C_out = 64):
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding)(x)
    x = layers.Conv2D(C_out, kernel_size=kernel_size, strides=stride, dilation_rate=dilation)(x)
    return x

def Conv(x, kernel_size, padding, stride = 1, C_out = 64):
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding)(x)
    x = layers.Conv2D(C_out, kernel_size=kernel_size, strides=stride)(x)
    return x

def Identity(x):
    return x

def cell1(x, flag=1):
    x1 = Conv(x, 5, 2)
    out_x = x + x1
    if flag == 1:
        out_x = layers.UpSampling2D(interpolation = 'bilinear')(out_x)
    return out_x

def cell2(x, flag=1):
    x1 = DilConv(x, 3, 2, 2)
    out_x = x + x1
    if flag == 1:
        out_x = layers.UpSampling2D(interpolation = 'bilinear')(out_x)
    return out_x

def cell3(x, flag=1):
    x1 = Conv(x, 3, 1)
    out_x = x + x1
    if flag == 1:
        out_x = layers.UpSampling2D(interpolation = 'bilinear')(out_x)
    return out_x

def cell4(x, flag=1):
    x1 = DilConv(x, 5, 8, 4)
    out_x = x + x1
    if flag == 1:
        out_x = layers.UpSampling2D(interpolation = 'bilinear')(out_x)
    return out_x

def cell_fuse(x, flag=1):
    x1 = DilConv(x, 3, 2, 2)
    out_x = x + x1
    if flag == 1:
        out_x = layers.UpSampling2D(interpolation = 'bilinear')(out_x)
    return out_x

def NetWork(image_size, shape, C = 64):
    # input = Input(shape=(299,299,shape))
    input = Input(shape=(image_size,image_size,shape))
    size = input.shape[1:3]
    conv1, conv2, conv3, conv4, conv5 = InceptionV3(input)
    dsn1 = layers.Conv2D(C, 1)(conv1)
    dsn2 = layers.Conv2D(C, 1)(conv2)
    dsn3 = layers.Conv2D(C, 1)(conv3)
    dsn4 = layers.Conv2D(C, 1)(conv4)
    dsn5 = layers.Conv2D(C, 1)(conv5)
    c1 = cell1(dsn5)

    mm1 = layers.concatenate([c1, crop(dsn4, (0, 0)+ c1.shape[1:3])])
    mm1 = layers.Conv2D(C, 1)(mm1)
    d4_2 = layers.Activation('relu')(mm1)
    c2 = cell2(d4_2)

    mm2 = layers.UpSampling2D(interpolation= 'bilinear')(c1)
    mm2 = layers.concatenate([mm2, crop(dsn3, (0, 0) + mm2.shape[1:3])])
    mm2 = layers.Conv2D(C, 1)(mm2)
    d3_2 = layers.Activation('relu')(mm2)
    d3_2 = layers.concatenate([c2, crop(d3_2, (0, 0) + c2.shape[1:3])])
    d3_2 = layers.Conv2D(C, 1)(d3_2)
    d3_2 = layers.Activation('relu')(d3_2)
    c3 = cell3(d3_2)

    c4 = cell4(dsn2)

    d_fuse = tf.zeros_like(c3)
    d_fuse = crop(layers.UpSampling2D(interpolation="bilinear")(c2), (0,0) + d_fuse.shape[1:3]) + crop(c3, (0, 0) + c3.shape[1:3]) + crop(layers.MaxPool2D()(c4), (0, 0) + d_fuse.shape[1:3])
    d_fuse = cell_fuse(d_fuse)
    d_fuse = layers.Conv2D(1, 1)(d_fuse)
    d_fuse = layers.ZeroPadding2D(7)(d_fuse)
    sss = layers.Conv2D(1, 15)(d_fuse)

    out_fuse = crop(sss, (34, 34) + size)
    out = crop(layers.Conv2D(1, 1)(layers.UpSampling2D(size=(4, 4), interpolation = 'bilinear')(c2)), (34, 34) + size)

    out_fuse = layers.Activation('sigmoid')(out_fuse)
    out = layers.Activation('sigmoid')(out)
    model = Model(input, [out_fuse, out], name='inception_v3')
    # model.summary()
    return model
    
def _upsample_like(src,tar):
    # src = tf.image.resize(images=src, size=tar.shape[1:3], method= 'bilinear')
    h = int(tar.shape[1]/src.shape[1])
    w = int(tar.shape[2]/src.shape[2])
    src = layers.UpSampling2D((h,w),interpolation='bilinear')(src)
    return src

def REBNCONV(x,out_ch=3,dirate=1):
    # x = layers.ZeroPadding2D(1*dirate)(x)
    x = layers.Conv2D(out_ch, 3, padding = "same", dilation_rate=1*dirate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def RSU7(hx, mid_ch=12, out_ch=3):
    hxin = REBNCONV(hx, out_ch,dirate=1)

    hx1 = REBNCONV(hxin, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx1)

    hx2 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx2)

    hx3 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx3)

    hx4 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx4)

    hx5 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx5)

    hx6 = REBNCONV(hx, mid_ch,dirate=1)

    hx7 = REBNCONV(hx6, mid_ch,dirate=2)

    hx6d = REBNCONV(layers.concatenate([hx7,hx6]), mid_ch,dirate=1)
    hx6dup = _upsample_like(hx6d,hx5)

    hx5d = REBNCONV(layers.concatenate([hx6dup,hx5]), mid_ch,dirate=1)
    hx5dup = _upsample_like(hx5d,hx4)

    hx4d = REBNCONV(layers.concatenate([hx5dup,hx4]), mid_ch,dirate=1)
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = REBNCONV(layers.concatenate([hx4dup,hx3]), mid_ch,dirate=1)
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = REBNCONV(layers.concatenate([hx3dup,hx2]), mid_ch,dirate=1)
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = REBNCONV(layers.concatenate([hx2dup,hx1]), out_ch,dirate=1)

    return hx1d + hxin

def RSU6(hx, mid_ch=12, out_ch=3):
    hxin = REBNCONV(hx, out_ch,dirate=1)
    
    hx1 = REBNCONV(hxin, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx1)

    hx2 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx2)

    hx3 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx3)

    hx4 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx4)

    hx5 = REBNCONV(hx, mid_ch,dirate=1)

    hx6 = REBNCONV(hx, mid_ch,dirate=2)


    hx5d =  REBNCONV(layers.concatenate([hx6,hx5]), mid_ch,dirate=1)
    hx5dup = _upsample_like(hx5d,hx4)

    hx4d = REBNCONV(layers.concatenate([hx5dup,hx4]), mid_ch,dirate=1)
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = REBNCONV(layers.concatenate([hx4dup,hx3]), mid_ch,dirate=1)
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = REBNCONV(layers.concatenate([hx3dup,hx2]), mid_ch,dirate=1)
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = REBNCONV(layers.concatenate([hx2dup,hx1]), out_ch,dirate=1)

    return hx1d + hxin

def RSU5(hx, mid_ch=12, out_ch=3):
    hxin = REBNCONV(hx, out_ch,dirate=1)

    hx1 = REBNCONV(hxin, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx1)

    hx2 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx2)

    hx3 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx3)

    hx4 = REBNCONV(hx, mid_ch,dirate=1)

    hx5 = REBNCONV(hx4, mid_ch,dirate=2)

    hx4d = REBNCONV(layers.concatenate([hx5,hx4]), mid_ch,dirate=1)
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = REBNCONV(layers.concatenate([hx4dup,hx3]), mid_ch,dirate=1)
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = REBNCONV(layers.concatenate([hx3dup,hx2]), mid_ch,dirate=1)
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = REBNCONV(layers.concatenate([hx2dup,hx1]), out_ch,dirate=1)

    return hx1d + hxin


def RSU4(hx,mid_ch=12, out_ch=3):
    hxin = REBNCONV(hx, out_ch,dirate=1)

    hx1 = REBNCONV(hxin,mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx1)

    hx2 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx2)

    hx3 = REBNCONV(hx, mid_ch,dirate=1)

    hx4 = REBNCONV(hx3, mid_ch,dirate=2)
    hx3d = REBNCONV(layers.concatenate([hx4,hx3]), mid_ch,dirate=1)
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = REBNCONV(layers.concatenate([hx3dup,hx2]), mid_ch,dirate=1)
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = REBNCONV(layers.concatenate([hx2dup,hx1]), out_ch,dirate=1)

    return hx1d + hxin

def RSU4F(hx, mid_ch=12, out_ch=3):
    hxin = REBNCONV(hx,out_ch,dirate=1)

    hx1 = REBNCONV(hxin, mid_ch,dirate=1)
    hx2 = REBNCONV(hx1, mid_ch,dirate=2)
    hx3 = REBNCONV(hx2, mid_ch,dirate=4)

    hx4 = REBNCONV(hx3, mid_ch,dirate=8)

    hx3d = REBNCONV(layers.concatenate([hx4,hx3]), mid_ch,dirate=4)
    hx2d = REBNCONV(layers.concatenate([hx3d,hx2]), mid_ch,dirate=2)
    hx1d = REBNCONV(layers.concatenate([hx2d,hx1]), out_ch,dirate=1)

    return hx1d + hxin

def U2NET(hx, out_ch=1):
    # hx = Input(shape=(480,480,3))
    #stage 1
    hx1 = RSU7(hx, 32,64)
    hx = layers.MaxPool2D(2,strides=2)(hx1)

    #stage 2
    hx2 = RSU6(hx, 32,128)
    hx = layers.MaxPool2D(2,strides=2)(hx2)

    #stage 3
    hx3 = RSU5(hx, 64,256)
    hx = layers.MaxPool2D(2,strides=2)(hx3)

    #stage 4
    hx4 = RSU4(hx, 128,512)
    hx = layers.MaxPool2D(2,strides=2)(hx4)

    #stage 5
    hx5 = RSU4F(hx, 256,512)
    hx = layers.MaxPool2D(2,strides=2)(hx5)

    #stage 6
    hx6 = RSU4F(hx, 256,512)
    hx6up = _upsample_like(hx6,hx5)

    #-------------------- decoder --------------------
    hx5d = RSU4F(layers.concatenate([hx6up,hx5]), 256,512)
    hx5dup = _upsample_like(hx5d,hx4)

    hx4d = RSU4(layers.concatenate([hx5dup,hx4]), 128,256)
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = RSU5(layers.concatenate([hx4dup,hx3]), 64,128)
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = RSU6(layers.concatenate([hx3dup,hx2]), 32,64)
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = RSU7(layers.concatenate([hx2dup,hx1]), 16,64)


    #side output
    d1 = layers.Conv2D(1, 3,padding="same")(hx1d)

    d2 = layers.Conv2D(1, 3,padding="same")(hx2d)
    d2 = _upsample_like(d2,d1)

    d3 = layers.Conv2D(1, 3,padding="same")(hx3d)
    d3 = _upsample_like(d3,d1)

    d4 = layers.Conv2D(1, 3,padding="same")(hx4d)
    d4 = _upsample_like(d4,d1)

    d5 = layers.Conv2D(1, 3,padding="same")(hx5d)
    d5 = _upsample_like(d5,d1)

    d6 = layers.Conv2D(1, 3,padding="same")(hx6)
    d6 = _upsample_like(d6,d1)

    d0 = layers.Conv2D(out_ch,1)(layers.concatenate([d1,d2,d3,d4,d5,d6]))

    o1    = layers.Activation('sigmoid')(d1)
    o2    = layers.Activation('sigmoid')(d2)
    o3    = layers.Activation('sigmoid')(d3)
    o4    = layers.Activation('sigmoid')(d4)
    o5    = layers.Activation('sigmoid')(d5)
    o6    = layers.Activation('sigmoid')(d6)
    ofuse = layers.Activation('sigmoid')(d0)

    return [ofuse, o1, o2, o3, o4, o5, o6]

def U2NETP(hx, out_ch=1):
    # hx = Input(shape=(480,480,3))
    #stage 1
    hx1 = RSU7(hx, 16,64)
    hx = layers.MaxPool2D(2,strides=2)(hx1)

    #stage 2
    hx2 = RSU6(hx, 16,64)
    hx = layers.MaxPool2D(2,strides=2)(hx2)

    #stage 3
    hx3 = RSU5(hx, 16,64)
    hx = layers.MaxPool2D(2,strides=2)(hx3)

    #stage 4
    hx4 = RSU4(hx, 16,64)
    hx = layers.MaxPool2D(2,strides=2)(hx4)

    #stage 5
    hx5 = RSU4F(hx, 16,64)
    hx = layers.MaxPool2D(2,strides=2)(hx5)

    #stage 6
    hx6 = RSU4F(hx, 16,64)
    hx6up = _upsample_like(hx6,hx5)

    #-------------------- decoder --------------------
    hx5d = RSU4F(layers.concatenate([hx6up,hx5]), 16,64)
    hx5dup = _upsample_like(hx5d,hx4)

    hx4d = RSU4(layers.concatenate([hx5dup,hx4]), 16,64)
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = RSU5(layers.concatenate([hx4dup,hx3]), 16,64)
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = RSU6(layers.concatenate([hx3dup,hx2]), 16,64)
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = RSU7(layers.concatenate([hx2dup,hx1]), 16,64)


    #side output
    d1 = layers.Conv2D(1, 3,padding="same")(hx1d)

    d2 = layers.Conv2D(1, 3,padding="same")(hx2d)
    d2 = _upsample_like(d2,d1)

    d3 = layers.Conv2D(1, 3,padding="same")(hx3d)
    d3 = _upsample_like(d3,d1)

    d4 = layers.Conv2D(1, 3,padding="same")(hx4d)
    d4 = _upsample_like(d4,d1)

    d5 = layers.Conv2D(1, 3,padding="same")(hx5d)
    d5 = _upsample_like(d5,d1)

    d6 = layers.Conv2D(1, 3,padding="same")(hx6)
    d6 = _upsample_like(d6,d1)

    d0 = layers.Conv2D(out_ch,1)(layers.concatenate([d1,d2,d3,d4,d5,d6]))

    o1    = layers.Activation('sigmoid')(d1)
    o2    = layers.Activation('sigmoid')(d2)
    o3    = layers.Activation('sigmoid')(d3)
    o4    = layers.Activation('sigmoid')(d4)
    o5    = layers.Activation('sigmoid')(d5)
    o6    = layers.Activation('sigmoid')(d6)
    ofuse = layers.Activation('sigmoid')(d0)

    return tf.stack([ofuse, o1, o2, o3, o4, o5, o6])

def Model_U2Net(image_size, num_band):
    hx = Input(shape=(image_size,image_size,num_band))
    out = U2NET(hx)
    model = Model(inputs = hx, outputs = out)
    return model

def Model_U2Netp(image_size, num_band):
    hx = Input(shape=(image_size,image_size,num_band))
    out = U2NETP(hx)
    model = Model(inputs = hx, outputs = out)
    return model

def UnetConv2(inputs, filters, kernel_size = (3, 3), kernel_initializer = "he_normal", strides = (1, 1), n = 2, name = ''):
    Z = inputs
    for i in range(1, n+1):
        Z = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides=strides, padding = "same",
                        kernel_initializer = kernel_initializer, name = f"{name}_unetconv2_conv{i}")(Z)
        Z = layers.BatchNormalization(axis = -1, name = f"{name}_unetconv2_bn{i}")(Z)
        Z = layers.Activation('relu', name=f"{name}_unetconv2_relu{i}")(Z)
    return Z

def Unet3plus_deep_supervision(inputs, upsample_size, filters = 1, kernel_size = (3, 3), kernel_initializer = "he_normal", strides = (1, 1), name = ''):
    Z = inputs
    Z = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides=strides, padding = "same",
                    kernel_initializer = kernel_initializer, name = f"{name}_ds_conv")(Z)

    Z = layers.UpSampling2D(size = upsample_size, interpolation = 'bilinear', name = f"{name}_ds_bilinear_upsample")(Z)
    Z = layers.Activation('sigmoid', dtype= 'float32', name=f"{name}_sigmoid")(Z)
    return Z

def FullScaleBlock(inputs, num_smaller_scale, filters = 64, kernel_size = (3, 3), kernel_initializer = "he_normal", strides = (1, 1), first_conv2d_strides = (1, 1), name = ''):
    inp_d = inputs
    for i in range(num_smaller_scale):
        inp_d[i] = layers.MaxPool2D(pool_size = 2 ** (num_smaller_scale - i), name = f"{name}_fsb_maxpool_{i}")(inp_d[i])
        inp_d[i] = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same',  
                                 kernel_initializer = kernel_initializer, name = f"{name}_fsb_conv_{i}")(inp_d[i])
        inp_d[i] = layers.BatchNormalization(axis = -1, name = f"{name}_fsb_bn_{i}")(inp_d[i])
        inp_d[i] = layers.ReLU(name = f"{name}_fsb_relu_{i}")(inp_d[i])

    inp_d[num_smaller_scale] = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same',
                                             kernel_initializer = kernel_initializer, name = f"{name}_fsb_conv_{num_smaller_scale}")(inp_d[num_smaller_scale])
    inp_d[num_smaller_scale] = layers.BatchNormalization(axis = -1, name = f"{name}_fsb_bn_{num_smaller_scale}")(inp_d[num_smaller_scale])
    inp_d[num_smaller_scale] = layers.ReLU(name = f"{name}_fsb_relu_{num_smaller_scale}")(inp_d[num_smaller_scale])

    for i in range(num_smaller_scale + 1, 5):
        inp_d[i] = layers.UpSampling2D(size = 2 ** (i - num_smaller_scale), interpolation = 'bilinear', name = f"{name}_fsb_bilinear_upsample_{i}")(inp_d[i])
        inp_d[i] = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same',
                                 kernel_initializer = kernel_initializer, name = f"{name}_fsb_conv_{i}")(inp_d[i])
        inp_d[i] = layers.BatchNormalization(axis = -1, name = f"{name}_fsb_bn_{i}")(inp_d[i])
        inp_d[i] = layers.ReLU(name = f"{name}_fsb_relu_{i}")(inp_d[i])
        
    Z = layers.Concatenate(axis = -1, name = f"{name}_fsb_concat")(inp_d)
    Z = layers.Conv2D(filters = 320, kernel_size = kernel_size, strides = strides, padding = 'same',
                     kernel_initializer = kernel_initializer, name = f"{name}_fsb_fusion_conv")(Z)
    Z = layers.BatchNormalization(axis = -1, name = f"{name}_fusion_bn")(Z)
    Z = layers.ReLU(name = f"{name}_fusion_relu")(Z)

    Zs = layers.Conv2D(filters = 1, kernel_size = (3, 3), strides = strides, padding = 'same',
                        kernel_initializer = kernel_initializer, name = f"{name}_fsb_sd_conv")(Z)
    
    if num_smaller_scale != 0:
        Zs = layers.UpSampling2D(size = 2 ** num_smaller_scale, interpolation = 'bilinear', name = f"{name}_fsb_sd_bilinear_upsample")(Zs)
    Zs = layers.Activation('sigmoid', dtype = 'float32', name = f"{name}_sd_sigmoid")(Zs)
    
    return Z, Zs


def Unet3plus(inputs, kernel_initializer = "he_normal", encoder_conv_n = 2):
    Z = inputs
    ## conv1    
#     Z1 = UnetConv2(Z, 64, name = 'conv1', kernel_size = (3, 3), kernel_initializer = "he_normal", strides = (1, 1), n = encoder_conv_n)
    Z1 = RSU7(Z, 16,32)
    Zo = layers.MaxPool2D(pool_size = 2, name = f"conv1_maxpool")(Z1)
    
    ## conv2    
#     Z2 = UnetConv2(Zo, 128, name = 'conv2', kernel_size = (3, 3), kernel_initializer = "he_normal", strides = (1, 1), n = encoder_conv_n)
    Z2 = RSU6(Zo, 32,64)
    Zo = layers.MaxPool2D(pool_size = 2, name = f"conv2_maxpool")(Z2)
    
    ## conv3    
#     Z3 = UnetConv2(Zo, 256, name = 'conv3', kernel_size = (3, 3), kernel_initializer = "he_normal", strides = (1, 1), n = encoder_conv_n)
    Z3 = RSU5(Zo, 64,128)
    Zo = layers.MaxPool2D(pool_size = 2, name = f"conv3_maxpool")(Z3)
    
    ## conv4    
#     Z4 = UnetConv2(Zo, 512, name = 'conv4', kernel_size = (3, 3), kernel_initializer = "he_normal", strides = (1, 1), n = encoder_conv_n)
    Z4 = RSU4(Zo, 128,256)
    Zo = layers.MaxPool2D(pool_size = 2, name = f"conv4_maxpool")(Z4)
    
    ## conv5    
#     Zd5 = UnetConv2(Zo, 1024, name = 'conv5', kernel_size = (3, 3), kernel_initializer = "he_normal", strides = (1, 1), n = encoder_conv_n)
    Zd5 = RSU4F(Zo, 256,512)
    
    Zsd5 = Unet3plus_deep_supervision(Zd5, upsample_size = 2 ** 4, name = "enc5_ds")
    
    # CGM
#     Z = layers.SpatialDropout2D(rate = 0.5)(Zd5)
#     Z = layers.Conv2D(filters = 1, kernel_size = (1,1), strides = (1, 1), padding = 'same', kernel_initializer = kernel_initializer, name = f"cgm_conv")(Z)
#     Z = tfa.layers.AdaptiveMaxPooling2D(output_size = 1, name = "cgm_adaptive_pool")(Z)
#     Zco = layers.Activation('sigmoid', name = 'cgm_sigmoid', dtype='float32')(Z)
#     Zc = tf.where(Zco > 0.5, 1., 0.)
#     Zco = tf.squeeze(Zco, axis = [-3, -2, -1], name = "Zco_squeeze")
#     Zsd5 = tf.multiply(Zsd5, Zc, name = "enc5_multiply")
    
    ## dec4
    Zd4, Zsd4 = FullScaleBlock([Z1, Z2, Z3, Z4, Zd5], num_smaller_scale = 3, name = "dec4")
#     Zsd4 = tf.multiply(Zsd4, Zc, name = "enc4_multiply")

    ## dec3
    Zd3, Zsd3 = FullScaleBlock([Z1, Z2, Z3, Zd4, Zd5], num_smaller_scale = 2, name = "dec3") 
#     Zsd3 = tf.multiply(Zsd3, Zc, name = "enc3_multiply")
    
    ## dec2
    Zd2, Zsd2 = FullScaleBlock([Z1, Z2, Zd3, Zd4, Zd5], num_smaller_scale = 1, name = "dec2") 
#     Zsd2 = tf.multiply(Zsd2, Zc, name = "enc2_multiply")
    
    ## dec1
    Zd1, Zsd1 = FullScaleBlock([Z1, Zd2, Zd3, Zd4, Zd5], num_smaller_scale = 0, name = "dec1") 
#     Zsd1 = tf.multiply(Zsd1, Zc, name = "enc1_multiply")
    
    return Zsd1, Zsd2, Zsd3, Zsd4, Zsd5

def Model_UNet3plus(image_size, num_band):
    hx = Input(shape=(image_size,image_size,num_band))
    out = Unet3plus(hx)
    model = Model(inputs = hx, outputs = out)
    # model.summary()
    return model

def IoULoss_(calc_axis = (-3, -2, -1), smooth = 1e-8):
    def IoULoss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        intersection = tf.reduce_sum(tf.multiply(y_true, y_pred), axis = calc_axis)
        total = tf.reduce_sum(y_true, axis = calc_axis) + tf.reduce_sum(y_pred, axis = calc_axis)
        union = total - intersection       
        IoU = (intersection + smooth) / (union + smooth)   
        return 1 - IoU
    return IoULoss

# class MS_SSIM_Loss(keras.losses.Loss):
#     def __init__(self, calc_axis = (-3, -2, -1), max_val = 1.0, **kwargs):
#         super().__init__(**kwargs)
        
#         self.calc_axis = calc_axis
#         self.max_val = max_val
        
        
#     def call(self, y_true, y_pred):
#         y_pred = tf.convert_to_tensor(y_pred)
#         y_true = tf.cast(y_true, y_pred.dtype)
        
#         ms_ssim = tf.image.ssim_multiscale(y_true, y_pred, max_val = self.max_val)
        
#         return 1 - ms_ssim

def pre_process_binary_cross_entropy(label, inputs):
    # preprocess data
    y = label
    loss = 0
    w_loss=1.0
    tmp_y = tf.cast(y, dtype=tf.float32)
    mask = tf.dtypes.cast(tmp_y > 0., tf.float32)
    b,h,w,c=mask.get_shape()
    positives = tf.math.reduce_sum(mask, axis=[1, 2, 3], keepdims=True)
    negatives = h*w*c-positives

    beta2 = positives / (negatives + positives) # negatives in hed
    beta = negatives / (positives + negatives) # positives in hed
    pos_w = tf.where(tf.equal(y, 0.0), beta2, beta)
    for tmp_p in inputs:
        l_cost = bce(y_true=tmp_y, y_pred=tmp_p, sample_weight=pos_w)
        loss += (l_cost*w_loss)
    return loss

def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def weighted_cross_entropy_loss(label, inputs):
    loss = 0
    y = tf.cast(label, dtype=tf.float32)
    negatives = tf.math.reduce_sum(1.-y)
    positives = tf.math.reduce_sum(y)
    beta = negatives/(negatives + positives)
    pos_w = beta/(1-beta)

    for predict in inputs:
        _epsilon = _to_tensor(tf.keras.backend.epsilon(), predict.dtype.base_dtype)
        predict   = tf.clip_by_value(predict, _epsilon, 1 - _epsilon)
        predict   = tf.math.log(predict/(1 - predict))
        
        cost = tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=predict, pos_weight=pos_w)
        cost = tf.math.reduce_mean(cost*(1-beta))
        loss += tf.where(tf.equal(positives, 0.0), 0.0, cost)
    return loss

def train_step(image_data, target):
    with tf.GradientTape() as tape:
        pred = my_model(image_data, training=True)
        loss = pre_process_binary_cross_entropy(target, pred)

    gradients = tape.gradient(loss, my_model.trainable_variables)
    del tape
    optimizer.apply_gradients(zip(gradients, my_model.trainable_variables))
    global_steps.assign_add(1)
    with writer.as_default():
        tf.summary.scalar("loss/loss", loss, step=global_steps)
    writer.flush()
        
    return loss.numpy()

def val_step(image_data, target):
    pred = my_model(image_data, training=False)
    loss = pre_process_binary_cross_entropy(target, pred)
    return loss.numpy()

if __name__ == '__main__':
    bce = tf.keras.losses.BinaryCrossentropy()
#     my_model = DexiNed(480, 3)
    my_model = Model_U2Net(480, 3)
#     my_model = Model_UNet3plus(480, 3)
    # my_model = NetWork(480, 3)
    # my_model.load_weights('/mnt/Nam/tmp_Nam/Nam_work_space/model/adalsn_farm_v3.h5')

    path1 = glob.glob('/mnt/Nam/tmp_Nam/Nam_work_space/data_train/wajo_image_z11_1706_999/*.npy')
    np.random.shuffle(path1)
    path2 = glob.glob('/mnt/Nam/tmp_Nam/Nam_work_space/data_train/image_40/*.npy')
    np.random.shuffle(path2)
    path3 = glob.glob('/mnt/Nam/tmp_Nam/Nam_work_space/data_train/image_train/*.npy')
    np.random.shuffle(path3)
    # path4 = glob.glob('/mnt/Nam/tmp_Nam/Nam_work_space/data_train/image_update/*.npy')
    # np.random.shuffle(path4)

    # path3 = glob.glob('/mnt/Nam/tmp_Nam/Nam_work_space/model_malay/image_update/*.npy')
    # np.random.shuffle(path3)
    
    alpha = 0.8
    # data_train = 5*path1[:int(len(path1)*alpha)]+11*path2[:int(len(path2)*alpha)]+path3[:int(len(path3)*alpha)]+path4[:int(len(path4)*alpha)]
    # data_val = 5*path1[int(len(path1)*alpha):]+11*path2[int(len(path2)*alpha):]+path3[int(len(path3)*alpha):]+path4[int(len(path4)*alpha):]  
    data_train = 3*path1[:int(len(path1)*alpha)]+9*path2[:int(len(path2)*alpha)]+path3[:int(len(path3)*alpha)]
    data_val = 3*path1[int(len(path1)*alpha):]+9*path2[int(len(path2)*alpha):]+path3[int(len(path3)*alpha):]
    # data_train = path3[:int(len(path3)* alpha)]
    # data_val = path3[int(len(path3)* alpha):]
    
    np.random.shuffle(data_train)
    np.random.shuffle(data_val)
    traindata = DataParser(data_train)
    valdata = DataParser(data_val)
    len_train = len(traindata)
    len_val = len(valdata)

    TRAIN_LOGDIR = '/mnt/Nam/tmp_Nam/Nam_work_space/logs'
    TRAIN_EPOCHS = 20
    best_val_loss = 1
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)


    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'GPUs {gpus}')
    if len(gpus) > 0:
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: pass
    if os.path.exists(TRAIN_LOGDIR): 
        try: shutil.rmtree(TRAIN_LOGDIR)
        except: pass
        
    writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
    validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    for epoch in range(TRAIN_EPOCHS):
        print('Epoch: %s / %s'%(str(epoch+1),str(TRAIN_EPOCHS)))
        with tqdm(total=len_train,desc=f'Train',postfix=dict,mininterval=0.3) as pbar:
            total_train = 0
            for image_data, target in traindata:
                results = train_step(image_data, target)
                total_train += results
                pbar.set_postfix(**{'total_loss' : results})
                pbar.update(1)              
            pbar.set_postfix(**{'total_train' : total_train/len_train})
            pbar.update(1)    
                 
        with tqdm(total=len_val,desc=f'Val',postfix=dict,mininterval=0.3) as pbar:
            total_val = 0
            for image_data, target in valdata:
                results = val_step(image_data, target)
                total_val += results
                pbar.set_postfix(**{'total_val' : results})
                pbar.update(1)
            pbar.set_postfix(**{'total_val' : total_val/len_val})
            pbar.update(1)
            with validate_writer.as_default():
                tf.summary.scalar("validate_loss/total_val", total_val/len_val, step=epoch)
            validate_writer.flush()
            
            if best_val_loss>=total_val/len_val:
                my_model.save_weights(os.path.join("/mnt/Nam/tmp_Nam/Nam_work_space/model", f"xxxxx.h5"))
                best_val_loss = total_val/len_val

        print(44*'-'+21*'*'+44*'-')
        print() 
