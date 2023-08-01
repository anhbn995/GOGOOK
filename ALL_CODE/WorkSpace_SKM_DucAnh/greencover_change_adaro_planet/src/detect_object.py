import rasterio
import threading
import numpy as np
import warnings, os
from tqdm import tqdm
import tensorflow as tf
import concurrent.futures
from rasterio.windows import Window

from model import unet_basic
from utils import get_range_value_planet_vung_miner
from tensorflow.compat.v1.keras.backend import set_session


warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))


DICT_COLORMAP =  {
                    0: (0,0,0, 0), # Nodata
                    1: (0,255,0,0), # Green
                    2: (100, 149, 237, 0), # water
                    3: (101,67,33, 0), # BuildUp
                    4: (0,128,0, 0), # Forest
                    5: (255,255,255,0) # Cloud
                }


def predict(model, path_image, path_predict, size=128):
    print(path_image)
    # qt_scheme = get_quantile_schema(path_image)
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
                    values = raster.read(window=read_wd)
                if raster.profile["dtype"]=="uint8":
                    # print('zo'*10, 'uint8')
                    image_detect = values[0:4].transpose(1,2,0).astype(int)
                else:
                    # datas = []
                    # for chain_i in range(4):
                    #     # band_qt = qt_scheme[chain_i]
                    #     band = values[chain_i]

                    #     # cut_nor = np.interp(band, (band_qt.get('p2'), band_qt.get('p98')), (1, 255)).astype(int)
                    #     cut_nor = get_range_value(band)
                    #     datas.append(cut_nor)
                    print('join')
                    datas = get_range_value_planet_vung_miner(values)
                    image_detect = np.transpose(datas, (1,2,0))

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

                if np.count_nonzero(image_detect) > 0:
                    if len(np.unique(image_detect)) <= 2:
                        pass
                    else:
                        y_pred = model.predict(image_detect[np.newaxis,...])
                        # print(time.time())
                        y_pred = (y_pred[0,...,0] > 0.5).astype(np.uint8) 
                        
                        # y_pred = 1 - y_pred
                        y = y_pred[mask].reshape(shape)
                        
                        with write_lock:
                            r.write(y[np.newaxis,...], window=Window(start_x, start_y, shape[1], shape[0]))
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                results = list(tqdm(executor.map(process, list_coordinates), total=len(list_coordinates)))



def get_index_nodata(in_fp_img):
    with rasterio.open(in_fp_img) as src:
        red = src.read(1)
        green = src.read(2)
        blue = src.read(3)
        nir = src.read(4)
        nodata_value = src.nodata

        red_mask = red == nodata_value
        green_mask = green == nodata_value
        blue_mask = blue == nodata_value
        nir_mask = nir == nodata_value

        # Kết hợp các mask lại với nhau
        combined_mask = red_mask & green_mask & blue_mask & nir_mask
        return np.where(np.array([combined_mask])==True)


def predict_each_class(in_fp, cnn_model, in_model_green, in_model_water, out_dir):
    out_green = os.path.join(out_dir, os.path.basename(in_fp).replace('.tif', '_green.tif'))
    out_water = os.path.join(out_dir, os.path.basename(in_fp).replace('.tif', '_water.tif'))

    cnn_model.load_weights(in_model_green)
    predict(cnn_model, in_fp, out_green, size=512)
    cnn_model.load_weights(in_model_water)
    predict(cnn_model, in_fp, out_water, size=512)
    return out_green, out_water


def union_class(in_fp, in_fp_out_green, in_fp_out_water, out_fp_union):
    ind_cloud = None
    ind_nodata = get_index_nodata(in_fp)

     # MOSAIC RS
    with rasterio.open(in_fp_out_water) as src_water:
        img_water = src_water.read()
        meta = src_water.meta
    ind_water = np.where(img_water != 0)
    
    # mosaic green vs water
    with rasterio.open(in_fp_out_green) as src_green:
        img_green = src_green.read()
    ind_green = np.where(img_green != 0)
    mask_green_water = np.zeros_like(img_water)

    mask_green_water[ind_green] = 1
    mask_green_water[ind_water] = 2
    mask_green_water[mask_green_water==0] = 3
    mask_green_water[ind_nodata] = 0
    if ind_cloud:
        mask_green_water[ind_cloud] = 5
    
    with rasterio.open(out_fp_union, 'w', **meta) as dst:
        dst.write(mask_green_water)
        dst.write_colormap(1, DICT_COLORMAP)


if __name__=="__main__":
    in_fp = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/img_ori/clip/2023-06_mosaic_cog_0.tif'
    in_model_green = r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Green_UINT8_4band_cut_512_stride_200_20230428_152806_V2_green.h5'
    in_model_water = r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Water_UINT8_4band_cut_512_stride_200_20230429_101659_V2_water.h5'
    out_dir = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/img_ori/clip/RS_TEST_XOA_3857'
    out_fp_union = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/img_ori/clip/RS_TEST_XOA_3857/2023-06_mosaic_union_ok.tif'
    os.makedirs('/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/img_ori/clip/RS_TEST_XOA_3857', exist_ok=True)
    size_model = 512
    cnn_model = unet_basic((size_model, size_model, 4))
    in_fp_out_green, in_fp_out_water, = predict_each_class(in_fp, cnn_model, in_model_green, in_model_water, out_dir)
    union_class(in_fp, in_fp_out_green, in_fp_out_water, out_fp_union)
    
    