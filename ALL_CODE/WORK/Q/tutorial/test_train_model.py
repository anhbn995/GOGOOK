# from tutorial import tmp
# from preprocess import tmp
import sys
sys.path.append('../')
# sys.path.insert(1, '/home/skm/SKM/WORK/ALL_CODE/WORK/Q')
# sys.path.insert(0,'/home/skm/SKM/WORK/ALL_CODE/WORK/Q/preprocess')
# sys.path.insert(0,'/home/skm/.vscode/extensions/ms-toolsai.jupyter-2022.9.1202862440/pythonFiles')
# sys.path.append('../')

import os
import json
from osgeo import gdal
import tensorflow as tf
from scripts.train import norm_train
# from models.metrics import iou, dice_coef
from models.callback.save_best import SavebestweightsandEarlyStopping

from models.loss import *
from models.core import *
from models.metrics import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, CategoricalAccuracy

# Setup giới hạn vram sử dụng làm hạn chếviệc tràn vram khi trainning
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4800)])
    except RuntimeError as e:
        print(e)
import os
import json
import tensorflow as tf

from scripts.train import norm_train
from models.metrics import iou, dice_coef
from models.callback.save_best import SavebestweightsandEarlyStopping

from models.loss import *
from models.core import *
from models.metrics import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, CategoricalAccuracy

# Setup giới hạn vram sử dụng làm hạn chếviệc tràn vram khi trainning
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4800)])
    except RuntimeError as e:
        print(e)

def main(mission, use_model, img_path, shp_path, box_path, old_weights):
    
    # Đọc dữ liệu prameters từ file json
    root_dir = os.path.dirname(sys.path[0]) 
    config_path = os.path.join(root_dir, 'configs', '%s.json'%(use_model))
    dict_params = json.load(open(config_path))

    total_stragegy = {}
    total_stragegy.update(dict_params['data'])
    total_stragegy.update(dict_params['strategy'])
    del dict_params['data'], dict_params['strategy'], dict_params['predict']
    
    use_model = dict_params['name']
    init_model = eval(use_model)
    model = init_model(**dict_params)
    # model.load_weights('/home/quyet/DATA_ML/WorkSpace/segmentation/weights/att_unet/att_unet_forest_monitor_2_512_5class_train.h5')

    losses_func = []
    for i in total_stragegy['losses']:
        losses_func.append(eval(i))

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
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only= True, 
                                                                    monitor='val_loss', mode='min', save_best_only=True)
    model_lrscheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=1)
    model_lrreduce_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience, min_lr=1e-7, verbose=1)
    model_earlystopping_callback = SavebestweightsandEarlyStopping(patience=patience, weights_path=weights_path)
    model_endtrainnan_callback = tf.keras.callbacks.TerminateOnNaN()
    model_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
    model_callbacks = [model_checkpoint_callback, model_lrscheduler_callback,
                        model_lrreduce_callback, model_earlystopping_callback,
                        model_tensorboard_callback,]

    history = norm_train(model, optimizer, losses_func, total_stragegy['loss_weights'], model_callbacks , model_metrics, 
                         data_path, img_path, shp_path, box_path, total_stragegy['img_size'], total_stragegy['num_band'], 
                         total_stragegy['epochs'], total_stragegy['batch_size'], dict_params['n_labels'], total_stragegy['split_ratios'], 
                         total_stragegy['use_test'], total_stragegy['use_multi'], old_weights, total_stragegy["img_size"], total_stragegy["stride_crop"])
    return history





# # # Khai báo tên task đang làm và tên model để lưu weights
# mission = 'road_multi'
# # mission = 'add_mangrove'
# use_model = 'swin_unet' # greencover sử dụng att_unet


# Duc Anh
# Khai báo tên task đang làm và tên model để lưu weights
mission = 'cloud_distribute_ducanh123aa'
# mission = 'add_mangrove'
use_model = 'att_unet' # greencover sử dụng att_unet

# Đường dẫn ảnh , box, label trong đó có các file phải cùng tên nhau và khác thư mục
# Data tạo ra ra được lưu trong đường dẫn /home/quyet/DATA_ML/WorkSpace/segmentation/data/ + tên mission 
# Cấu trúc data train gồm 2 thư mục :
# ---img_cut_crop: *.tif
# ---mask_cut_crop: *.tif

img_path = '/home/skm/SKM16/Data/ZZ/img'
shp_path = '/home/skm/SKM16/Data/ZZ/label'
box_path = '/home/skm/SKM16/Data/ZZ/box'
old_weights = None

# img_path = '/home/quyet/DATA_ML/WorkSpace/segmentation/tutorial/LuyenAll/images'
# shp_path = '/home/quyet/DATA_ML/WorkSpace/segmentation/tutorial/LuyenAll/lables_water'
# box_path = '/home/quyet/DATA_ML/WorkSpace/segmentation/tutorial/LuyenAll/boxs'
# old_weights = '/home/quyet/DATA_ML/WorkSpace/segmentation/tutorial/LuyenAll/water_weights.h5'
print(img_path, '/n', shp_path, '/n', box_path)
history = main(mission, use_model, img_path, shp_path, box_path, old_weights)