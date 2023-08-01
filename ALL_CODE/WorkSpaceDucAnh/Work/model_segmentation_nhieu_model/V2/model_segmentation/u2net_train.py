from tensorflow.keras import layers, Model, regularizers, Input
import tensorflow as tf
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2
# import rasterio
import shutil
from tensorflow.compat.v1.keras.backend import set_session
import os, glob, shutil, warnings

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# "GPU"
warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))

# VisEff = VisualEffect()
class DataParser():
    def __init__(self, annotation):
        self.total_data = annotation
        self.batch_size = 2
        self.steps_per_epoch = int(len(self.total_data)//self.batch_size)
        self.augmentations = [self.flip_ud, self.flip_lr, self.rot90]
        
    def generate_minibatches(self):
        while True:
            np.random.shuffle(self.total_data)
            for idx in range(self.steps_per_epoch):
                filename = self.total_data[idx*self.batch_size:(idx+1)*self.batch_size]
                image, label = self.get_batch(filename)
                yield image, 7*[label]


    def get_batch(self, batch):
        images = []
        edgemaps = []
        for img_list in batch:
            im = np.load(img_list).transpose(1,2,0)[...,:3]
            em = np.load(img_list.replace('npy_i_256', 'npy_m_256')).transpose(1,2,0)
            # im = VisEff(im)
            im = np.array(im/255.0, dtype=np.float32)
            em = np.array(em/255.0, dtype=np.float32)
            # em = em.astype(np.float32)

            for f in self.augmentations:
                if np.random.uniform()<0.20:
                    im, em=f(im, em)

            images.append(im)
            edgemaps.append(em)

        images   = np.asarray(images)
        edgemaps = np.asarray(edgemaps)

        return images, edgemaps

    
    def flip_ud(self, im, em):
        return np.flipud(im), np.flipud(em)

    def flip_lr(self, im, em):
        return np.fliplr(im), np.fliplr(em)

    def rot90(self, im, em):
        return np.rot90(im), np.rot90(em)
    
    def __len__(self):
        return self.steps_per_epoch
   

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
    # print(x)
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

def U2NET(hx, in_ch=3,out_ch=1):
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

    o1    = layers.Activation('sigmoid', name="o1")(d1)
    o2    = layers.Activation('sigmoid', name="o2")(d2)
    o3    = layers.Activation('sigmoid', name="o3")(d3)
    o4    = layers.Activation('sigmoid', name="o4")(d4)
    o5    = layers.Activation('sigmoid', name="o5")(d5)
    o6    = layers.Activation('sigmoid', name="o6")(d6)
    ofuse = layers.Activation('sigmoid', name="ofuse")(d0)

    return [o1, o2, o3, o4, o5, o6, ofuse]

def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def weight_cross_entropy(y_true, y_pred):
    _epsilon = _to_tensor(tf.keras.backend.epsilon(), y_pred.dtype.base_dtype)
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.math.log(y_pred/ (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.math.reduce_sum(1. - y_true)
    count_pos = tf.math.reduce_sum(y_true)

    beta = count_neg / (count_neg + count_pos)
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)
    cost = tf.reduce_mean(cost * (1 - beta))
    
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)

def pixel_accuracy(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    acc = tf.cast(tf.equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(acc, name='pixel_accuracy')

def Model_U2Net(image_size, num_band):
    hx = Input(shape=(image_size,image_size,num_band))
    out = U2NET(hx)
    model = Model(inputs = hx, outputs = out)
    model.compile(loss={"o1": weight_cross_entropy,
                        "o2": weight_cross_entropy,
                        "o3": weight_cross_entropy,
                        "o4": weight_cross_entropy,
                        "o5": weight_cross_entropy,
                        "o6": weight_cross_entropy,
                        "ofuse": weight_cross_entropy,
        },
                  metrics= pixel_accuracy)
    return model

if __name__== "__main__":
    
    my_model = Model_U2Net(256, 3)
    path3 = glob.glob('/home/skm/SKM_OLD/ZZ_ZZ/model_segmentation/V2/Data_Train_V2/256/npy_i_256/*.npy')
    # print(path3)
    np.random.shuffle(path3)
    alpha=0.8
    data_train = path3[:int(len(path3)* alpha)]
    data_val = path3[int(len(path3)* alpha):]
    traindata = DataParser(data_train)
    valdata = DataParser(data_val)
    
    # for i,j in valdata.generate_minibatches():
    #     print('------')
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='logs_u2net_256_ok', histogram_freq=0, write_graph=True, write_images=False),
        tf.keras.callbacks.ModelCheckpoint(os.path.join('logs_u2net_256_oke', "model{epoch:02d}.h5"), verbose=0, save_weights_only=True, save_best_only = True),
    ]
        
    my_model.fit_generator(
        traindata.generate_minibatches(),
        epochs=100,
        callbacks=callbacks,
        steps_per_epoch = len(traindata),
        validation_data=valdata.generate_minibatches(),
        validation_steps= len(valdata)
    )