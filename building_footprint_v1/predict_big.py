from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
import rasterio
import numpy as np
import time
import os
import torch
import torch.nn as nn

import cv2
from tqdm import *
import sys
import glob
import logging

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

@torch.no_grad()
def predict(img_url, threshold = 0.5, cnn_model=None):
    num_band = 4

    batch_size = 4
    input_size = 300
    INPUT_SIZE = 300

    dataset = gdal.Open(img_url)
    values = dataset.ReadAsArray()[0:num_band].astype(np.float16)
    ## Crop attributes
    h,w = values.shape[1:3]
    crop_size = 300 # input_size//2 # begin = 200
    padding = int((input_size - crop_size)/2)
    padded_org_im = []
    cut_imgs = []
    new_w = w + 2*padding
    new_h = h + 2*padding
    cut_w = list(range(padding, new_w - padding, crop_size))
    cut_h = list(range(padding, new_h - padding, crop_size))

    list_hight = []
    list_weight = []
    for i in cut_h:
        if i < new_h - padding - crop_size:
            list_hight.append(i)
    list_hight.append(new_h-crop_size-padding)

    for i in cut_w:
        if i < new_w - crop_size - padding:
            list_weight.append(i)
    list_weight.append(new_w-crop_size-padding)

    img_coords = []
    for i in list_weight:
        for j in list_hight:
            img_coords.append([i, j])
    
    for i in range(num_band):
        # print(values[i], padding)
        band = np.pad(values[i], padding, mode='reflect')
        padded_org_im.append(band)
    # values = np.pad(values, padding, mode='reflect',axis=-1)

    values = np.array(padded_org_im).swapaxes(0,1).swapaxes(1,2)
    del padded_org_im

    # def preprocess_val(img):
    #     ######### Histogram Equalization
    #     # print(img.shape)                
    #     transform = A.CLAHE(p = 1)
    #     agumented = transform(image=img)
    #     img = agumented['image']
    #     ################################
    #     return img

    def get_im_by_coord(org_im, start_x, start_y,num_band):
        startx = start_x-padding
        endx = start_x+crop_size+padding
        starty = start_y - padding
        endy = start_y+crop_size+padding
        # pnt
        # img = [
        #     org_im[i][starty:endy, startx:endx ] for i in range(num_band)
        #     # org_im[3][starty:endy, startx:endx],
        # ]
        result=[]
        img = org_im[starty:endy, startx:endx]
        img = img.swapaxes(2,1).swapaxes(1,0)
        for chan_i in range(num_band):
            result.append(img[chan_i])     #cv2.resize(img[chan_i],(INPUT_SIZE, INPUT_SIZE), interpolation = cv2.INTER_CUBIC)
        return np.array(result).swapaxes(0,1).swapaxes(1,2)

    for i in range(len(img_coords)):
        im = get_im_by_coord(
            values, img_coords[i][0], img_coords[i][1],num_band)
        # im = im[...,:3]
        # im2 = preprocess_grayscale(im)
        # im2 = normalize_4band_esri(im, num_band)
        # im2 = normalize_255(im)
        # im2 = normalize_bandcut(im, num_band)
        cut_imgs.append(im)
    # print(len(cut_imgs))

    a = list(range(0, len(cut_imgs), batch_size))

    if a[len(a)-1] != len(cut_imgs):
        a[len(a)-1] = len(cut_imgs)
    
    y_pred = []
    for i in tqdm(range(len(a)-1)):
        x_batch = []
        x_batch = np.array(cut_imgs[a[i]:a[i+1]], dtype=np.float16)/np.finfo(np.float16).max

        y_batch = cnn_model.predict(x_batch)
        y_batch = np.array(y_batch.squeeze(3))
        y_pred.extend(y_batch)
    big_mask = np.zeros((h, w)).astype(np.float16)
    for i in range(len(cut_imgs)):
        if np.count_nonzero(cut_imgs[i]) > 0:
            true_mask = y_pred[i].reshape((INPUT_SIZE,INPUT_SIZE))
            true_mask = (true_mask > threshold).astype(np.uint8)
        else:
            true_mask = np.zeros((INPUT_SIZE,INPUT_SIZE))
        true_mask = (cv2.resize(true_mask,(input_size, input_size), interpolation = cv2.INTER_CUBIC)).astype(np.uint8)
        start_x = img_coords[i][1]
        start_y = img_coords[i][0]
        big_mask[start_x-padding:start_x-padding+crop_size, start_y-padding:start_y -
                    padding+crop_size] += true_mask[padding:padding+crop_size, padding:padding+crop_size]
        # x = big_mask[start_x-padding:start_x-padding+crop_size, start_y-padding:start_y -
        #             padding+crop_size]
        # print('hello')
        # if any(x) > 1:
        #     print(x)
        #     sys.exit(0)

    del cut_imgs
    big_mask = (big_mask ).astype(np.uint8)
    return big_mask*255


def predict_all(base_path,outputFileName, model, postprocess=False):
    from frame_field_learning.inference_from_filepath import inference_from_filepath
    # Predict output mask
    mask_base = predict(base_path, threshold=0.5, cnn_model=model)
    ################################

    with rasterio.open(base_path) as src:
        transform1 = src.transform
        w,h = src.width,src.height
        crs = src.crs
    
    new_dataset = rasterio.open(outputFileName, 'w', driver='GTiff',
                                height = h, width = w,
                                count=1, dtype="uint8",
                                crs=crs,
                                transform=transform1,
                                compress='lzw')
    new_dataset.write(mask_base, 1)
    new_dataset.close()

if __name__ == '__main__':
    # img_data = r"/home/geoai/eodata/cloud_detect_tensorflow/raw_img/cloud/sentinel_1.tif"
    # img_data = r"/home/geoai/eodata/cloud_detect_tensorflow/cloud_only/data/S2A_MSIL2A_20220217T222551_N0400_R029_T59GPQ_20220221T000951.tif"
    
    img_data = r"/home/geoai/eodata/cloud_detect_tensorflow/raw_final/predict/S2A_MSIL2A_20221002T031551_N0400_R118_T48NUG_20221002T074557_haze_41.tif"
    
    list_img = [img_data]
    for img_data in list_img:
        output_shp = img_data.replace(".tif", "_masked.tif")
        # FN_MODEL = r"/home/skymap/data/building_extraction_torch/frame_field_learning/runs/mapping_dataset.unet_resnet101_pretrained | 2022-11-23 17:34:05/checkpoints/checkpoint.epoch_000001.tar"
        
        # Define model & load weights
        cnn_model = build_model((None,None,4), 42)
        #############################
        logging.info("Loading model {}".format(cnn_model.name))
        cnn_model.load_weights(FN_MODEL)
        logging.info("Model loaded !")
        
        predict_all(img_data, output_shp, model=cnn_model, postprocess=False)
        print('Process successfully, output saved at', output_shp)
