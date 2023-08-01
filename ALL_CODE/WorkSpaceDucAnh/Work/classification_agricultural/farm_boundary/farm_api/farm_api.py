# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 11:24:51 2022

@author: AnhHo
"""
from flask_restful import reqparse, abort, Api, Resource
from flask import request, abort, Flask
from threading import Thread
import rasterio
import numpy as np
from rasterio.windows import Window
from tqdm import tqdm
import warnings, cv2, os
import tensorflow as tf
import Vectorization
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects
from tensorflow.compat.v1.keras.backend import set_session
from PIL import Image
import uuid
import json
warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))

app = Flask(__name__)
api = Api(app)
app_port = 6790

def get_unique_id():
    id = str(uuid.uuid4()).replace("-","")
    return id

def predict_farm(model, path_image, path_predict, size=480):
    with rasterio.open(path_image) as raster:
        meta = raster.meta
        meta.update({'count': 1, 'nodata': 0})
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
                
        with tqdm(total=len(list_coordinates)) as pbar:
            with rasterio.open(path_predict,'w+', **meta, compress='lzw') as r:
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
                            y_pred = model.predict(image_detect[np.newaxis,...]/255.)[0]
                            y_pred = (y_pred[0,...,0] > 0.5).astype(np.uint8)
                            y = y_pred[mask].reshape(shape)
                            r.write(y[np.newaxis,...], window=Window(start_x, start_y, shape[1], shape[0]))
                    pbar.update()

def Morphology(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # dilation  
    # img = cv2.dilate(data,kernel,iterations = 1)
    # opening
    #     img = cv2.morphologyEx(data, cv2.MORPH_OPEN, kernel)
    # for i in range(10):
    #     img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2)
    # closing
    #     for _ in range(2):
    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
    return img
    
def raster_to_vector(path_in, path_out, threshold_distance, threshold_connect):
    print('start convert raster to vector ...')
    with rasterio.open(path_in) as inds:
        data = inds.read()[0]
        transform = inds.transform
        projstr = inds.crs.to_string()
        
    data = Morphology(data)
    data = remove_small_holes(data.astype(bool), area_threshold=77)
    data = remove_small_objects(data, min_size=77)
    skeleton = skeletonize(data.astype(np.uint8))
    
    Vectorization.save_polygon(np.pad(skeleton, pad_width=1).astype(np.intc), threshold_distance, threshold_connect, transform, projstr, path_out)
    print("Done!!!")

model_path = './model_farm.h5'
size = 480
threshold_distance = 3 #ngưỡng làm mượt polygon
threshold_connect = 5 #ngưỡng nối điểm gần nhau
model_farm = tf.keras.models.load_model(model_path)

class Upload_tiles(Resource):
    def post(self):
        file = request.files['file']
        left = request.form['left']
        top = request.form['top']
        right = request.form['right']
        bottom = request.form['bottom']
        zoom_level = request.form['zoom_level']
        print(zoom_level)
        img = Image.open(file)
        img = np.array(img)
        img = img.transpose(2,0,1)
        transform=rasterio.transform.from_bounds(float(left), float(bottom), float(right), float(top), img.shape[2], img.shape[1])
        crs = rasterio.crs.CRS({"init": "epsg:4326"})
        file_name=get_unique_id()
        file_path = './upload/{}.tif'.format(file_name)
        output_path = './upload/{}_predict.geojson'.format(file_name)
        cache_path = output_path.replace('.geojson', '.tif')
        with rasterio.open(
                file_path,
                'w',
                driver='GTiff',
                height=img.shape[1],
                width=img.shape[2],
                count=3,
                dtype=img.dtype,
                crs=crs,
                transform=transform,
            ) as dst:
            dst.write(img[0], 1)
            dst.write(img[1], 2)
            dst.write(img[2], 3)
        predict_farm(model_farm, file_path, cache_path, size)
        raster_to_vector(cache_path, output_path, threshold_distance, threshold_connect)
        os.remove(cache_path)

        print(transform)
        return json.load(open(output_path))



if __name__ == '__main__':
    # t1 = Thread(target=cal_farm_process, args=())
    # t1.daemon = True
    # t1.start()

    api.add_resource(Upload_tiles, '/submit')
    app.run(debug=0,host="0.0.0.0",port=app_port)
