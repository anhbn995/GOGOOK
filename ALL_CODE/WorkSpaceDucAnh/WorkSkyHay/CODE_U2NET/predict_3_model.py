import threading
import rasterio
import numpy as np
import warnings, os
from tqdm import tqdm
import tensorflow as tf
import concurrent.futures
from rasterio.windows import Window
from tensorflow.compat.v1.keras.backend import set_session

warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))
num_bands = 3
size = 512
def get_quantile_schema(img):
    pass


def predict_farm(model, path_image, path_predict, size=512):
    qt_scheme = get_quantile_schema(path_image)
    with rasterio.open(path_image) as raster:
        meta = raster.meta
        meta.update({'count': 1, 'nodata': 0,"dtype":"uint8"})
        height, width = raster.height, raster.width
        input_size = size
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
        with rasterio.open(path_predict,'w+', **meta, compress='lzw') as r:
            read_lock = threading.Lock()
            write_lock = threading.Lock()

            def process(coordinates):
                x_off, y_off, x_count, y_count, start_x, start_y = coordinates
                read_wd = Window(x_off, y_off, x_count, y_count)
                with read_lock:
                    values = raster.read(window=read_wd)[0:num_bands]
                if raster.profile["dtype"]=="uint8":
                    # print('vao')
                    image_detect = values.transpose(1,2,0).astype(int)
                    
                else:
                    datas = []
                    for chain_i in range(3):
                        band_qt = qt_scheme[chain_i]
                        band = values[chain_i]
                        cut_nor = np.interp(band, (band_qt.get('p2'), band_qt.get('p98')), (1, 255)).astype(int)
                        datas.append(cut_nor)
                    image_detect = np.array(datas).transpose(1,2,0)
                img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
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

                    image_detect = img_temp
                mask = (mask!=0)
                
                if not np.all(image_detect == 0):
                    y_pred = model.predict(image_detect[np.newaxis,...]/255.)
                    y_pred = np.array(y_pred)
                    y_pred = (y_pred[0,0,...,0] > 0.5).astype(np.uint8)
                    y = y_pred[mask].reshape(shape)
                    with write_lock:
                        r.write(y[np.newaxis,...], window=Window(start_x, start_y, shape[1], shape[0]))
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                results = list(tqdm(executor.map(process, list_coordinates), total=len(list_coordinates)))


if __name__=="__main__":
    
    model_path_chanel = r'/home/skm/SKM16/Data/IIIII/Data_Train_Chanel/Chanel_512_add_fix/logs/u2net_512_chanel_V1_fix_model.h5'
    model_path_pipeline = r'/home/skm/SKM16/Data/IIIII/Data_Train_Pipeline/Pipeline_512/logs/u2net_512_Pipeline_V0_model.h5'
    model_path_pond = r'/home/skm/SKM16/Data/IIIII/Data_Train_Pond_fix/Pond_512_fix/logsfix/u2net_512_pond_V0_model.h5'
    
    # fp_in = r'/home/skm/SKM16/Data/IIIII/Image_all/Data_oke/images7.tif'
    # fp_out = r'/home/skm/SKM16/Data/IIIII/Image_all/Data_oke/i7_pond.tif'
    # dir_tmp = r'/home/skm/SKM16/Data/IIIII/Image_all/Data_oke/xx_pond'
    
    
    fp_in = r'/home/skm/SKM16/Data/IIIII/Image_all/Data_oke/images7.tif'
    fp_out = r'/home/skm/SKM16/Data/IIIII/Image_all/Data_oke/i7_pond_green.tif'
    dir_tmp = r'/home/skm/SKM16/Data/IIIII/Image_all/Data_oke/xx_pond_xoa'
    
    os.makedirs(dir_tmp, exist_ok=True)
    
    os.makedirs(os.path.dirname(fp_out), exist_ok=True)
    list_fp_model = [model_path_pond, model_path_pipeline, model_path_chanel]
    
    i = 0
    for fp_model in list_fp_model:
        i+=1
        model = tf.keras.models.load_model(fp_model)
        output_tmp_path = os.path.join(dir_tmp,os.path.basename(fp_out).replace('.tif',f'_{str(i)}.tif'))
        predict_farm(model, fp_in, output_tmp_path, size)

   