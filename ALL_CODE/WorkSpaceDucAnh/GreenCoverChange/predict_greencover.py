import numpy as np
import rasterio
from rasterio.windows import Window
import threading
from tqdm import tqdm
import concurrent.futures
import warnings, cv2, os
import tensorflow as tf

from utils import *
from tensorflow.compat.v1.keras.backend import set_session


warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))

def predict(model, path_image, path_predict, size=128):
    print(path_image)
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
                    values = raster.read(window=read_wd)[0:4]
                if raster.profile["dtype"]=="uint8":
                    # print('zo'*10, 'uint8')
                    image_detect = values[0:4].transpose(1,2,0).astype(int)
                else:
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
                        y_pred = (y_pred[0,...,0] > 0.9).astype(np.uint8)
                        y = y_pred[mask].reshape(shape)
                        
                        with write_lock:
                            r.write(y[np.newaxis,...]*255, window=Window(start_x, start_y, shape[1], shape[0]))
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                results = list(tqdm(executor.map(process, list_coordinates), total=len(list_coordinates)))


def Morphology(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
    return img
    
    
if __name__=="__main__":
    
    from predict_config import *
    list_image=create_list_id(folder_image_path)
    folder_output_path_water = os.path.join(folder_output_path, "water09")
    folder_output_path_green = os.path.join(folder_output_path, "green")
    
    model.load_weights(model_path)
    if not os.path.exists(folder_output_path_water):
        os.makedirs(folder_output_path_water)
    
    for image_path in tqdm(list_image):
        image_name = os.path.basename(image_path)     
        fp_img = os.path.join(folder_image_path, image_name)+".tif"
        outputpredict = os.path.join(folder_output_path_water,image_name) +".tif"
        
        if not os.path.exists(outputpredict):
            print(fp_img)
            predict(model, fp_img, outputpredict, size_model)
        else:
            pass
    
    # from predict_config import *
    # list_name_dir = ["AOI_1","AOI_2","AOI_3","AOI_7","AOI_8","AOI_9","AOI_10","AOI_11","AOI_12","AOI_14","AOI_15"]
    # for name_dir in list_name_dir:
    #     folder_image_path = os.path.join(folder_image_path_ori, name_dir)
    #     list_image=create_list_id(folder_image_path)
    #     folder_output_path_water = os.path.join(os.path.join(folder_output_path, name_dir), "water09")
    #     folder_output_path_green = os.path.join(os.path.join(folder_output_path, name_dir), "green")
        
    #     model_water.load_weights(model_path_water)
    #     if not os.path.exists(folder_output_path_water):
    #         os.makedirs(folder_output_path_water)
        
    #     for image_path in tqdm(list_image):
    #         image_name = os.path.basename(image_path)     
    #         fp_img = os.path.join(folder_image_path, image_name)+".tif"
    #         outputpredict = os.path.join(folder_output_path_water,image_name) +".tif"
            
    #         if not os.path.exists(outputpredict):
    #             print(fp_img)
    #             predict(model_water, fp_img, outputpredict, size_model)
    #         else:
    #             pass
        
        """greem"""
        # model_green.load_weights(model_path_green)    
        # if not os.path.exists(folder_output_path_green):
        #     os.makedirs(folder_output_path_green)
        
        # for image_path in tqdm(list_image):
        #     image_name = os.path.basename(image_path)     
        #     fp_img = os.path.join(folder_image_path, image_name)+".tif"
        #     outputpredict = os.path.join(folder_output_path_green,image_name) +".tif"
            
        #     if not os.path.exists(outputpredict):
        #         print(fp_img)
        #         predict(model_green, fp_img, outputpredict, size_model)
        #     else:
        #         pass