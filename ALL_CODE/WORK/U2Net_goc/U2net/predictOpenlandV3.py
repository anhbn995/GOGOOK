#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 09:26:05 2021

@author: ducanh
"""
# import cv2
import rasterio
import numpy as np
from tqdm import tqdm
import os, glob, warnings
from rasterio.windows import Window
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session

warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))


num_chanel = 3
input_size = 512
INPUT_MOEL = 512
crop_size = 300

def write_window_many_chanel(output_ds, arr_c, s_h, e_h ,s_w, e_w, sw_w, sw_h, size_w_crop, size_h_crop):
    # for c, arr in enumerate(arr_c):
    # print(arr_c.shape, 'zzzzzz')
    output_ds.write(arr_c[0][s_h:e_h,s_w:e_w], window = Window(sw_w, sw_h, size_w_crop, size_h_crop), indexes=1)
        

def predict_small(cnn_model, image_detect, dict_min_max=None):
    image_detect = image_detect.transpose(1,2,0)
    y_pred = cnn_model.predict(image_detect[np.newaxis,...]/255.)[0]
    y_pred = (y_pred[0,...,0] > 0.5).astype(np.uint8)
    y_pre = y_pred.reshape((INPUT_MOEL,INPUT_MOEL))
    # print(np.array([y_pre]), "nnnnnnn")
    return np.array([y_pre])


def predict_big(image_path, outputFileName, cnn_model):
    dict_min_max = None

    with rasterio.open(image_path) as src:
        h,w = src.height,src.width
        source_crs = src.crs
        source_transform = src.transform
        # dtype_or = src.dtypes
        num_band = src.count - 1

    # create img predict one band 
    with rasterio.open(outputFileName, 'w', driver='GTiff',
                                height = h, width = w,
                                count=1, dtype='uint8',
                                crs=source_crs,
                                transform=source_transform,
                                nodata=0,
                                compress='lzw') as output_ds:
        output_ds = np.empty((1,h,w), dtype='uint8')
        
    padding = int((input_size - crop_size)/2)
    list_weight = list(range(0, w, crop_size))
    list_hight = list(range(0, h, crop_size))

    with rasterio.open(image_path) as src:
        with rasterio.open(outputFileName,"r+") as output_ds:
            with tqdm(total=len(list_hight)*len(list_weight)) as pbar:
                for start_h_org in list_hight:
                    for start_w_org in list_weight:
                        # vi tri bat dau
                        h_crop_start = start_h_org - padding
                        w_crop_start = start_w_org - padding
                        # kich thuoc
                        tmp_img_size_model = np.zeros((num_band, input_size,input_size))
                        # truong hop 0 0
                        if h_crop_start < 0 and w_crop_start < 0:
                            # continue
                            h_crop_start = 0
                            w_crop_start = 0
                            size_h_crop = crop_size + padding
                            size_w_crop = crop_size + padding
                            img_window_crop  = src.read(window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))[:3]
                            tmp_img_size_model[:, padding:, padding:] = img_window_crop
                            # print(np.unique(tmp_img_size_model), np.unique(img_window_crop), img_window_crop.shape)
                            if not np.all(tmp_img_size_model==0):
                                img_predict = predict_small(cnn_model, tmp_img_size_model, dict_min_max)
                                write_window_many_chanel(output_ds, img_predict, padding, crop_size + padding, padding, crop_size + padding, 
                                                                            start_w_org, start_h_org, crop_size, crop_size)
                        
                        # truong hop h = 0 va w != 0
                        elif h_crop_start < 0:
                            h_crop_start = 0
                            size_h_crop = crop_size + padding
                            size_w_crop = min(crop_size + 2*padding, w - start_w_org + padding)
                            img_window_crop  = src.read(window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))[:3]
                            
                            if size_w_crop == w - start_w_org + padding:
                                end_c_index_w =  size_w_crop
                                tmp_img_size_model[:,padding:,:end_c_index_w] = img_window_crop
                            else:
                                end_c_index_w = crop_size + padding
                                tmp_img_size_model[:, padding:,:] = img_window_crop
                            # print(np.unique(tmp_img_size_model), np.unique(img_window_crop), img_window_crop.shape)
                            if not np.all(tmp_img_size_model==0):
                                img_predict = predict_small(cnn_model, tmp_img_size_model, dict_min_max)
                                write_window_many_chanel(output_ds, img_predict, padding, crop_size + padding ,padding, end_c_index_w, 
                                                        start_w_org, start_h_org,  min(crop_size, w - start_w_org), crop_size)
                        
                        # Truong hop w = 0, h!=0 
                        elif w_crop_start < 0:
                            w_crop_start = 0
                            size_w_crop = crop_size + padding
                            size_h_crop = min(crop_size + 2*padding, h - start_h_org + padding)
                            img_window_crop  = src.read(window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))[:3]
                            
                            if size_h_crop == h - start_h_org + padding:
                                end_c_index_h =  size_h_crop
                                tmp_img_size_model[:,:end_c_index_h,padding:] = img_window_crop
                            else:
                                end_c_index_h = crop_size + padding
                                tmp_img_size_model[:,:, padding:] = img_window_crop
                            # print(np.unique(tmp_img_size_model), np.unique(img_window_crop), img_window_crop.shape)
                            if not np.all(tmp_img_size_model==0):
                                img_predict = predict_small(cnn_model, tmp_img_size_model, dict_min_max)
                                write_window_many_chanel(output_ds, img_predict, padding, end_c_index_h, padding, crop_size + padding, 
                                                        start_w_org, start_h_org, crop_size, min(crop_size, h - start_h_org))
                            
                        # Truong hop ca 2 deu khac khong
                        else:
                            size_w_crop = min(crop_size +2*padding, w - start_w_org + padding)
                            size_h_crop = min(crop_size +2*padding, h - start_h_org + padding)
                            img_window_crop  = src.read(window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))[:3]
                            # print(img_window_crop.shape, size_w_crop, size_h_crop)
                            if size_w_crop < (crop_size + 2*padding) and size_h_crop < (crop_size + 2*padding):
                                # print(img_window_crop.shape, size_w_crop, size_h_crop)
                                end_c_index_h = size_h_crop
                                end_c_index_w = size_w_crop
                                tmp_img_size_model[:,:end_c_index_h,:   end_c_index_w] = img_window_crop
                            elif size_w_crop < (crop_size + 2*padding):
                                end_c_index_h = crop_size + padding
                                end_c_index_w = size_w_crop
                                tmp_img_size_model[:,:,:end_c_index_w] = img_window_crop
                            elif size_h_crop < (crop_size + 2*padding):
                                end_c_index_w = crop_size + padding
                                end_c_index_h = size_h_crop
                                tmp_img_size_model[:,:end_c_index_h,:] = img_window_crop
                            else:
                                end_c_index_w = crop_size + padding
                                end_c_index_h = crop_size + padding
                                tmp_img_size_model[:,:,:] = img_window_crop
                            # print(np.unique(tmp_img_size_model), np.unique(img_window_crop), img_window_crop.shape)
                            if not np.all(tmp_img_size_model==0):
                                img_predict = predict_small(cnn_model, tmp_img_size_model, dict_min_max) 
                                write_window_many_chanel(output_ds, img_predict, padding, end_c_index_h, padding, end_c_index_w, 
                                                        start_w_org, start_h_org, min(crop_size, w - start_w_org), min(crop_size, h - start_h_org))
                        pbar.update()


def get_list_fp(folder_dir, type_file = '*.tif'):
        """
            Get all file path with file type is type_file.
        """
        list_fp = []
        for file_ in glob.glob(os.path.join(folder_dir, type_file)):
            head, tail = os.path.split(file_)
            list_fp.append(os.path.join(head, tail))
        return list_fp

def bo_file_tao_ra_muon_nhat_trong_list(list_fp):
    time_max = 0
    fp_break = 's'
    for fp in list_fp:
        time_create = os.path.getmtime(fp)
        if time_create > time_max:
            time_max = time_create
            fp_break = fp
    list_fp.remove(fp_break)
    return list_fp

def keep_list_fp_dont_have_list_eliminate(list_have_all, list_eliminate):
    list_eliminate = [os.path.basename(fp) for fp in list_eliminate]
    if list_eliminate:
        list_keep = []
        for fp in list_have_all:
            if os.path.basename(fp) not in list_eliminate:
                list_keep.append(fp)
        return list_keep
    else:
        return list_have_all

def main(dir_img, dir_out, model_path, check_runed = True):
    os.makedirs(dir_out, exist_ok=True)
    
    lines = [dir_img, dir_out, model_path]
    with open(os.path.join(dir_out, 'config.txt'), 'w') as f:
        for line in lines:
            f.write(f"{line}\n")
    
    cnn_model = tf.keras.models.load_model(model_path)     
    list_img = glob.glob(os.path.join(dir_img,'*.tif'))
    print(len(list_img), 'tat ca')
    
    if check_runed:
    # khong chay nhung anh chay r
        list_runed = glob.glob(os.path.join(dir_out,'*.tif'))
        if list_runed:
            list_runed = bo_file_tao_ra_muon_nhat_trong_list(list_runed)
            list_img = keep_list_fp_dont_have_list_eliminate(list_img, list_runed)
            print(len(list_runed), 'da chay')
            print(len(list_img), 'can chay')
        
        
    for input_path in tqdm(list_img):
        output_path = os.path.join(dir_out, os.path.basename(input_path))
        predict_big(input_path, output_path, cnn_model)
        # predict_farm(model_farm, input_path, output_path, size)
        
    
    
    
if __name__ == "__main__":
    # import time
    # x = time.time()
    # dir_img = r"C:\Users\SkyMap\Downloads\tmp\zzzzzzzzzzzzzzzzzz.tif"
    # dir_out = r"C:\Users\SkyMap\Downloads\tmp\a.tif"
    model_path = r'/home/skm/SKM16/ALL_MODEL/Openland/logs_Water_of_Openland_V2_1666753511/weight/Water_of_Openland_V2_1666753512_loadmodel.h5'
    # main(dir_img, dir_out, model_path, check_runed = True)
    # print(time.time()-x)
    
    # cnn_model = tf.keras.models.load_model(model_path)  
    # predict_big(dir_img, dir_out, cnn_model)
    dir_img = r"/home/skm/SKM16/Work/OpenLand/all_tif"
    dir_out = r"/home/skm/SKM16/Work/OpenLand/all_tif/Water_of_Openland_V2_1666753512"
    
    os.makedirs(dir_out, exist_ok=True)
    list_img = glob.glob(os.path.join(dir_img,'*.tif'))
    print(len(list_img), 'all')

    # khong chay nhung anh chay r
    # list_runed = glob.glob(os.path.join(dir_out,'*.tif'))
    # list_runed = bo_file_tao_ra_muon_nhat_trong_list(list_runed)
    # list_img = keep_list_fp_dont_have_list_eliminate(list_img, list_runed)
    # print(len(list_runed), 'da chay')
    # print(len(list_img), 'tru')
    # khong chay nhung anh chay r 

    

    cnn_model = tf.keras.models.load_model(model_path)
    for input_path in tqdm(list_img, desc=f'Run all {len(list_img)}'):
        output_path = os.path.join(dir_out, os.path.basename(input_path))
        print(os.path.basename(input_path))
        predict_big(input_path, output_path, cnn_model)
