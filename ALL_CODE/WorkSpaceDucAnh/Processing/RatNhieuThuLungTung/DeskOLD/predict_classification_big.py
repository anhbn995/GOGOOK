# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 07:53:31 2018

@author: ducanh
"""
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
import rasterio
from rasterio.transform import from_origin
import numpy as np
from keras import backend as K
import time
import model_classification, model_defores
# K.set_image_dim_ordering("th")
import tensorflow as tf
import os
import gdal
import cv2
from tqdm import *
import sys
import glob
import ogr
import osr
from lib.convert_datatype import list_contour_to_list_polygon, list_polygon_to_list_contour
from lib.export_data import exportResult2 as exportResult2
from matplotlib import pyplot as plt
# K.clear_session()
# FN_MODEL = r"/media/skymap/data/training_unkown/models/unet_basic/20190624_165630/20190624_165630_0020_val_weights.h5"
FN_MODEL = r"/mnt/D850AAB650AA9AB0/PADDY/models/unet_paddy_sentinel/20190913_112628/20190913_112628_0102_val_weights.h5"
FN_MODEL = r"/media/skymap/Backup/Blora_and_Grobogan/train/models/mazine_res_unet50_india/20200402_234454/20200402_234454_0041_val_weights.h5"
FN_MODEL = r"/media/skymap/Data/Lapan_gamma0_avg/tmp/models/unet50_paddy_lapan_avg/20200404_021148/20200404_021148_0008_val_weights.h5"
FN_MODEL = r"/media/skymap/Backup/Blora_and_Grobogan/train_3_to_6_2018/tmp/models/unet50_mazie_m3_m6_2018/20200408_164156/20200408_164156_0043_val_weights.h5"
FN_MODEL = r"/media/skymap/Backup/Blora_and_Grobogan/train_11to2_2017_2018/tmp/models/unet50_mazie_11to2_2017_2018/20200409_091951/20200409_091951_0059_val_weights.h5"
FN_MODEL = r"/media/skymap/Backup/Blora_and_Grobogan/train_7to10_2018/tmp/models/unet50_mazie_m7_m10_2018/20200409_111859/20200409_111859_0039_val_weights.h5"
FN_MODEL=r"/media/skymap/Data/Lapan_gamma0_avg/tmp/Image_cut_img_mask_crop/models/unet50_paddy_classify/20200417_091727/20200417_091727_0042_val_weights.h5"
FN_MODEL=r"/media/skymap/Data/block_building/New_Sample_data_TRIPLESAT_SOI/data_road_unet/models/unet50_road/20200430_105109/20200430_105109_0077_val_weights.h5"
num_chanel = 4
INPUT_SIZE = 256
num_class = 2
def predict(img_url, threshold = 0.5):
    num_band = 4
    # cnn_model = model.unet_mixed_conv(num_channel=num_band, size=512)
    # cnn_model = model_classification.unet_basic(num_channel=num_band, size=256)
    cnn_model = model_defores.unet_50(num_channel=num_band, size=256,num_class=num_class)
    cnn_model.load_weights(FN_MODEL)  
    batch_size = 32
    input_size = 256
    crop_size = 768//4
    dataset1 = gdal.Open(img_url)
    values = dataset1.ReadAsArray()[0:num_band].astype(np.uint8)
    h,w = values.shape[1:3]    
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
        band = np.pad(values[i], padding, mode='reflect')
        padded_org_im.append(band)
    # values = np.pad(values, padding, mode='reflect',axis=-1)

    values = np.array(padded_org_im).swapaxes(0,1).swapaxes(1,2)
    print(values.shape)
    del padded_org_im

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
        # print(starty,endy)
        img = img.swapaxes(2,1).swapaxes(1,0)
        for chan_i in range(num_chanel):
            result.append(cv2.resize(img[chan_i],(INPUT_SIZE, INPUT_SIZE), interpolation = cv2.INTER_CUBIC))
        return np.array(result).swapaxes(0,1).swapaxes(1,2)

    for i in range(len(img_coords)):
        im = get_im_by_coord(
            values, img_coords[i][0], img_coords[i][1],num_band)
        # print(im.shape)
        # im2 = preprocess_grayscale(im)
        # im2 = normalize_4band_esri(im, num_band)
        # im2 = normalize_255(im)
        # im2 = normalize_bandcut(im, num_band)
        cut_imgs.append(im)
    # print(len(cut_imgs))

    a = list(range(0, len(cut_imgs), batch_size))

    if a[len(a)-1] != len(cut_imgs):
        a[len(a)-1] = len(cut_imgs)

    # y_pred = []
    # # with tqdm(total=len(list_id)) as pbar:
    # for i in range(len(a)-1):
    #     x_batch = []
    #     x_batch = np.array(cut_imgs[a[i]:a[i+1]]).astype(np.float32)/255.0
    #     y_batch = cnn_model.predict(x_batch)
    #     # print(y_batch.shape)
    #     y_pred.extend(y_batch)
    big_mask = np.zeros((h, w, num_class)).astype(np.uint8)
    for i in range(len(cut_imgs)):
        y_pred = cnn_model.predict(np.array([cut_imgs[i]])/255.0)
        # print(y_pred[i].shape)
        # if i%5 == 0:
        #     plt.imshow(y_pred[i][:,:,[0,1,2]])
        #     plt.show()
        true_mask = (y_pred[0]).argmax(axis=2)

        # true_mask = ((y_pred[0])>=0.5).astype(np.uint8)
        seg_img = np.zeros( ( INPUT_SIZE , INPUT_SIZE , num_class ) )
        for c in range(num_class):
            seg_img[:,:,c] = (true_mask == c).astype(np.uint8)
        true_mask = seg_img.swapaxes(2,1).swapaxes(1,0)

        masks = []
        for chan_i in range(num_class):
            masks.append((cv2.resize(true_mask[chan_i],(input_size, input_size), interpolation = cv2.INTER_CUBIC)>0.5).astype(np.uint8))
        true_mask = np.array(masks).swapaxes(0,1).swapaxes(1,2)
        start_x = img_coords[i][1]
        start_y = img_coords[i][0]
        # print(start_x-padding, start_x-padding+crop_size, start_y-padding, start_y -
        #         padding+crop_size)
        big_mask[start_x-padding:start_x-padding+crop_size, start_y-padding:start_y -
                    padding+crop_size] = true_mask[padding:padding+crop_size, padding:padding+crop_size]

    del cut_imgs
    big_mask = (big_mask > 0.5).astype(np.uint8)*255
    return big_mask



def predict_all(base_path,outputFileName):
    mask_base = predict(base_path)
#     print(mask_base.shape)
#     # mask = mask_base[:,:,0]
    with rasterio.open(base_path) as src:
        transform1 = src.transform
        w,h = src.width,src.height
        projstr = (src.crs.to_string())
        print(projstr)
        crs = src.crs
        check_epsg = crs.is_epsg_code
        # if check_epsg:
        coordinate = src.crs.to_epsg()
    mask_result = np.zeros((h, w)).astype(np.uint8)
    mask_result[mask_base[:,:,0]==255] = 1
    mask_result[mask_base[:,:,1]==255] = 2
    # mask_result[mask_base[:,:,2]==255] = 3

####################################
    if coordinate!= None:
        crs = rasterio.crs.CRS.from_epsg(coordinate)
    else:
        crs = rasterio.crs.CRS.from_string(projstr)
    new_dataset = rasterio.open(outputFileName, 'w', driver='GTiff',
                                height = h, width = w,
                                count=2, dtype="uint8",
                                crs=crs,
                                transform=transform1,
                                compress='lzw')
# print(masking(r"/media/building/building/data_source/tmp/Malaysia-jupem/image_mask/forest.tif")[0].shape)
    new_dataset.write(mask_base[:,:,0],1)
    new_dataset.write(mask_base[:,:,1],2)
    # new_dataset.write(mask_base[:,:,2],3)
####################################

    # new_dataset = rasterio.open(outputFileName, 'w', driver='GTiff',
    #                             height = h, width = w,
    #                             count=1, dtype="uint8",
    #                             crs=crs,
    #                             transform=transform1,
    #                             nodata=0,
    #                             compress='lzw')
    # new_dataset.write(mask_result,1)
    # new_dataset.close()
    # im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # dataset = gdal.Open(base_path)
    # driverName = "ESRI Shapefile"
    
    # geotransform = dataset.GetGeoTransform()
    # projection = osr.SpatialReference(dataset.GetProjectionRef())
    # # print(projection)
    # polygons_result = list_contour_to_list_polygon(contours)
    # exportResult2(polygons_result, geotransform, projection, outputFileName, driverName)

if __name__ == '__main__':
    # return True
    predict_all(r"/media/skymap/Data/block_building/New_Sample_data_TRIPLESAT_SOI/image_8bit_perimage/Clip1_sample.tif",
    r"/media/skymap/Data/block_building/New_Sample_data_TRIPLESAT_SOI/image_8bit_perimage/Clip1_sample_rs60.tif")
    # test()
