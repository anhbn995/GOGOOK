# import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os, cv2
from tensorflow.keras import layers, backend, Model, utils
import tensorflow as tf
import glob, warnings
# from color_image import VisualEffect
from tensorflow.compat.v1.keras.backend import set_session

warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))

def BatchActivate(x):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
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
    optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(optimizer=optimizer,
                loss=losses,
                metrics=['accuracy', f1_m, precision_m, recall_m])

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

losses=tf.keras.losses.BinaryCrossentropy()
smooth = 1. # Used to prevent the denominator from 0.
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

class DataParser():
    def __init__(self, annotations, batch_size):
        self.total_data = annotations
        self.batch_size_train = batch_size
        self.steps_per_epoch = int(len(self.total_data)//self.batch_size_train)
        self.augmentations = [self.flip_ud, self.flip_lr, self.rot90, self.blur]

    def generate_minibatches(self):
        while True:
            np.random.shuffle(self.total_data)
            for i in range(self.steps_per_epoch):
                batch_ids=self.total_data[i*self.batch_size_train:(i+1)*self.batch_size_train]

                ims, ems = self.get_batch(batch_ids)
                yield(ims, ems)


    def get_batch(self, batch):
        images = []
        edgemaps = []
        for img_list in batch:
            im = np.load(img_list)
            em = np.load(img_list.replace('image', 'label'))
            im = np.array(im, dtype=np.float32)
            em = np.array(em, dtype=np.float32)
            
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

    def blur(self, im, em):
        return cv2.GaussianBlur(im,(5,5),0), em
    
    def __len__(self):
        return self.steps_per_epoch
    
if __name__=="__main__":
    # VisEff = VisualEffect()
    def lr_decay(epoch):
        initial_learningrate=0.001
        if epoch < 1:
            return initial_learningrate
        else:
            return initial_learningrate * 0.9 ** epoch
    callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='/home/skm/SKM16/ALL_MODEL/SolarPanel/weights', histogram_freq=0, write_graph=True, write_images=False),
    tf.keras.callbacks.ModelCheckpoint(os.path.join('/home/skm/SKM16/ALL_MODEL/SolarPanel/weights', "unet_bce_solarPanel.h5"), verbose=1, save_weights_only=True, save_best_only=True),
    tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=1),
    ]
    model = build_model((256,256,4), 32)

    path_train = "/home/skm/SKM16/Work/SonalPanel_ThaiLand/1Ver2_lable2/izmages_8bit_perizmage/Data_Train_and_Model/U2net_Ds/train/image/*.npy"
    path_val = "/home/skm/SKM16/Work/SonalPanel_ThaiLand/1Ver2_lable2/izmages_8bit_perizmage/Data_Train_and_Model/U2net_Ds/val/image/*.npy"
    batch_size = 4
    data_train = glob.glob(path_train)
    data_val = glob.glob(path_val)
    print(len(data_val))
    traindata = DataParser(data_train, batch_size)
    testdata = DataParser(data_val, batch_size)
    
    # model.load_weights('log/unet.h5')
    model.fit_generator(
        traindata.generate_minibatches(),
        epochs=100,
        callbacks=callbacks,
        steps_per_epoch = len(traindata),
        validation_data=testdata.generate_minibatches(),
        validation_steps= len(testdata)
    )