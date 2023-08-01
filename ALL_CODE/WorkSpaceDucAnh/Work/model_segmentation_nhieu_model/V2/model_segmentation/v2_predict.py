import numpy as np
import matplotlib.pyplot as plt
import rasterio
from models.import_module import DexiNed, Model_U2Netp, Model_U2Net, Adalsn, Model_UNet3plus
from rasterio.windows import Window
from tqdm import tqdm
import warnings, os
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session


warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))

model_path2 = '/media/skymap/Nam/tmp_Nam/Nam_work_space/model/model_u2net.h5'
model2 = Model_U2Net(480, 3)
model2.load_weights(model_path2)

# print("****")
# model2 = tf.keras.models.load_model('/mnt/data/eof/model_farm_boundary/model_farm.h5')

# input_path = '/mnt/data/farm_ggmap/image_label/image/b5_gmap.tif'
# output_path = '/mnt/data/model_segmentation/aaa.tif'

folder = "/media/skymap/Nam/public/farm_uint16/mosaic histo_8u.tif"
out_folder = '/media/skymap/Nam/public/farm_uint16/mosaic histo_8uxxxxxxx.tif'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)
    
name_image = os.listdir(folder)

# list_infor_run = [FN_MODEL]
for _file in name_image:
    input_path = folder + _file
    output_path = out_folder + _file
    if os.path.exists(output_path):
        continue
    with rasterio.open(input_path) as raster:
        meta = raster.meta
        meta.update({'count': 1, 'nodata': 0})
        height, width = raster.height, raster.width
        input_size = 480
        stride_size = input_size - input_size //4
        padding = int((input_size - stride_size) / 2)
        
        list_coordinates = []
        for start_y in range(0, height, stride_size):
            for start_x in range(0, width, stride_size): 
                x_off = start_x if start_x==0 else start_x - padding
                y_off = start_y if start_y==0 else start_y - padding
                    
                end_x = min(start_x + stride_size + padding, width)
                end_y = min(start_y + stride_size + padding, height)
                
                x_count = end_x - x_off
                y_count = end_y - y_off
                list_coordinates.append(tuple([x_off, y_off, x_count, y_count, start_x, start_y]))
                
        with tqdm(total=len(list_coordinates)) as pbar:
            with rasterio.open(output_path,'w+', **meta, compress='lzw') as r:
                for x_off, y_off, x_count, y_count, start_x, start_y in list_coordinates:
                    image_detect = raster.read(window=Window(x_off, y_off, x_count, y_count))[0:3].transpose(1,2,0)
                    mask = np.pad(np.ones((stride_size, stride_size), dtype=np.uint8), ((padding, padding),(padding, padding)))
                    shape = (stride_size, stride_size)
                    if y_count < input_size or x_count < input_size:
                        img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                        mask = np.zeros((input_size, input_size), dtype=np.uint8)
                        if start_x == 0 and start_y == 0:
                            img_temp[(input_size - y_count):input_size, (input_size - x_count):input_size] = image_detect
                            mask[(input_size - y_count):input_size-padding, (input_size - x_count):input_size-padding] = 1
                            shape = (y_count-padding, x_count-padding)
                        elif start_x == 0:
                            img_temp[0:y_count, (input_size - x_count):input_size] = image_detect
                            if y_count == input_size:
                                mask[padding:y_count-padding, (input_size - x_count):input_size-padding] = 1
                                shape = (y_count-2*padding, x_count-padding)
                            else:
                                mask[padding:y_count, (input_size - x_count):input_size-padding] = 1
                                shape = (y_count-padding, x_count-padding)
                        elif start_y == 0:
                            img_temp[(input_size - y_count):input_size, 0:x_count] = image_detect
                            if x_count == input_size:
                                mask[(input_size - y_count):input_size-padding, padding:x_count-padding] = 1
                                shape = (y_count-padding, x_count-2*padding)
                            else:
                                mask[(input_size - y_count):input_size-padding, padding:x_count] = 1
                                shape = (y_count-padding, x_count-padding)
                        else:
                            img_temp[0:y_count, 0:x_count] = image_detect
                            mask[padding:y_count, padding:x_count] = 1
                            shape = (y_count-padding, x_count-padding)
                            
                        # image_detect = np.array((img_temp/127.5) - 1, dtype=np.float32)
                        image_detect = img_temp
                    mask = (mask!=0)
                        
                    if np.count_nonzero(image_detect) > 0:
                        if len(np.unique(image_detect)) <= 2:
                            pass
                        else:
                            y_pred = model2.predict(image_detect[np.newaxis,...]/255.)[0]
                            y_pred = (y_pred[0,...,0] > 0.5).astype(np.uint8)
                            y = y_pred[mask].reshape(shape)
                            r.write(y[np.newaxis,...], window=Window(start_x, start_y, shape[1], shape[0]))
                    pbar.update()
        
    
    

