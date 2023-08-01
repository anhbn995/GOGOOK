import numpy as np
import matplotlib.pyplot as plt
import rasterio
from models.import_module import DexiNed, Model_U2Netp, Model_U2Net, Adalsn, Model_UNet3plus
import math, os
import warnings
from tqdm import tqdm
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# warnings.filterwarnings("ignore")
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
# set_session(tf.compat.v1.Session(config=config))

# """"""""""""
# model_path2 = os.path.join("/home/skm/SKM_OLD/ZZ_ZZ/model_segmentation/V2/model_segmentation/weights/VN/", f"u2net.h5")
# model2 = Model_U2Net(256, 3)
# model2.load_weights(model_path2)
# """"""""""""

# tf.keras.models.save_model(model2, '/mnt/data/model_farm_boundary/model_farm.h5')
# model2 = tf.keras.models.load_model('/mnt/data/model_farm_boundary/model_farm.h5')

model_path2 = os.path.join("/home/skm/SKM_OLD/ZZ_ZZ/cloud_shadow/Model_UNet3plus/data_train_bo_dagood/model_256_good_boda/", f"model_Adalsn_256_boda_good.h5")
model2 = Model_UNet3plus(256, 3)
model2.load_weights(model_path2)

import glob
def get_list_fp(folder_dir, type_file = '*.tif'):
        """
            Get all file path with file type is type_file.
        """
        list_fp = []
        for file_ in glob.glob(os.path.join(folder_dir, type_file)):
            # head, tail = os.path.split(file_)
            # list_fp.append(os.path.join(head, tail))
            list_fp.append(file_)
        return list_fp


list_fp = get_list_fp("/home/skm/SKM_OLD/ZZ_ZZ/NewImages", type_file = '*.tif')  
for path_image in list_fp:
    print(path_image)
    # path_image = "/home/skm/SKM_OLD/ZZ_ZZ/NewImages/DeGi.tif"
    name = os.path.basename(path_image)
    path_predict = os.path.join('/home/skm/SKM_OLD/ZZ_ZZ/model_segmentation/V2/BoTrain_U3net/moi', name)

    with rasterio.open(path_image) as raster:
        data = raster.read().transpose(1,2,0)
        height,width = raster.height, raster.width
        meta = raster.meta
        input_size = 256
        stride_size = input_size - input_size//4
        padding = int((input_size - stride_size)/2)
        
        padding_height = math.ceil((height+padding-input_size+stride_size)/stride_size)*stride_size+input_size-height-padding
        padding_width = math.ceil((width+padding-input_size+stride_size)/stride_size)*stride_size+input_size-width-padding
        
        data_pad = np.pad(data, ((padding, padding_height),(padding, padding_width), (0,0)))
        new_height, new_width = data_pad.shape[:2]
        y_predict = np.zeros((new_height, new_width), dtype=np.uint8)
        for y in tqdm(range(0, new_height, stride_size)):
            for x in range(0, new_width, stride_size):
                x_predict = data_pad[y:y+input_size, x:x+input_size,...]
                if np.count_nonzero(x_predict) > 0:
                    y_pred = model2.predict(x_predict[np.newaxis,...]/255.)[0]
                    print(np.unique(y_pred))
                    y_pred = (y_pred[0,...,0] > 0.5).astype(np.uint8)
                    y_predict[y+padding:y+input_size-padding, x+padding:x+input_size-padding] = y_pred[padding:input_size-padding, padding:input_size-padding]

        y = y_predict[padding:new_height-padding_height, padding:new_width-padding_width]
        meta.update({'count': 1, 'nodata': 0})
        with rasterio.open(path_predict, 'w', **meta, compress='lzw') as rrr:
            rrr.write(y[np.newaxis,...])