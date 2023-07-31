import os
import sys
import json
import argparse
import tensorflow as tf

from models.loss import *
from models.core import *
from models.metrics import *
from scripts.train import norm_train
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from models.metrics import iou, dice_coef
from models.callback.save_best import SavebestweightsandEarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall

gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#   except RuntimeError as e:
#     print(e)

if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4800)])
  except RuntimeError as e:
    print(e)

def main(mission, use_model, img_path, shp_path, box_path, old_weights):
    root_dir = sys.path[0]
    config_path = os.path.join(root_dir, 'configs', '%s.json'%(use_model))
    init_model = eval(use_model)
    dict_params = json.load(open(config_path))

    total_stragegy = {}
    total_stragegy.update(dict_params['data'])
    total_stragegy.update(dict_params['strategy'])
    del dict_params['data'], dict_params['strategy'], dict_params['predict']

    model = init_model(**dict_params)

    losses_func = []
    for i in total_stragegy['losses']:
        losses_func.append(eval(i)())

    model_metrics = []
    for j in total_stragegy['metrics']:
        model_metrics.append(eval(j))

    optimizer = eval(total_stragegy['optimizer'])

    print("Init callback function")
    def lr_decay(epoch):
        if epoch < 1:
            return total_stragegy['init_loss']
        else:
            return total_stragegy['init_loss'] * 0.98 ** (epoch)

    data_path = os.path.join(os.path.join(root_dir, 'data', mission))
    checkpoint_filepath = os.path.join(root_dir, 'logs', mission, 'tmp')
    log_dir = os.path.join(root_dir, 'logs', mission, 'graph')
    weights_path = os.path.join(root_dir, 'weights', '%s'%(use_model), use_model+'_'+mission+'_'+str(total_stragegy['img_size'])+'_'+str(dict_params['n_labels'])+'class.h5')
    patience = 10
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only= True, 
                                                                    monitor='val_loss', mode='min', save_best_only=True)
    lrscheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=1)
    lrreduce_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience, min_lr=1e-7, verbose=1)
    earlystopping_callback = SavebestweightsandEarlyStopping(patience=patience, weights_path=weights_path)
    endtrainnan_callback = tf.keras.callbacks.TerminateOnNaN()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
    model_callbacks = [checkpoint_callback, lrscheduler_callback,
                       lrreduce_callback, earlystopping_callback,
                       tensorboard_callback, endtrainnan_callback]

    history = norm_train(model, optimizer, losses_func, total_stragegy['loss_weights'], model_callbacks , model_metrics, 
                         data_path, img_path, shp_path, box_path, total_stragegy['img_size'], total_stragegy['num_band'], 
                         total_stragegy['epochs'], total_stragegy['batch_size'], dict_params['n_labels'], total_stragegy['split_ratios'], 
                         total_stragegy['use_test'], total_stragegy['use_multi'], old_weights, total_stragegy["img_size"], total_stragegy["stride_crop"])
    return history

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mission', type=str, default='forest', help='Define mission of model', required=False)
    parser.add_argument('--name_model', type=str, default='att_unet', help='Choose model to use', required=False)
    parser.add_argument('--img_path', type=str, default='/home/quyet/DATA_ML/Projects/Npark_forest/data_fo_train/img', 
                        help='Define path of folder contains images', required=False)
    parser.add_argument('--shp_path', type=str, default='/home/quyet/DATA_ML/Projects/Npark_forest/data_fo_train/label', 
                        help='Define path of folder contains labels', required=False)
    parser.add_argument('--box_path', type=str, default='/home/quyet/DATA_ML/Projects/Npark_forest/data_fo_train/box', 
                        help='Define path of folder contains boxes', required=False)
    parser.add_argument('--weight_path', type=str, default='/home/quyet/DATA_ML/Projects/segmentation/weights/att_unet/att_unet_forest_256_1class_train.h5', 
                        help='Define path of weight', required=False)

    args = parser.parse_args()
    main(args.mission, args.name_model, args.img_path, args.shp_path, args.box_path, args.weight_path)