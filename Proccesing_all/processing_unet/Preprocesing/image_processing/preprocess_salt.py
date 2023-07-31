import argparse
import math
import glob
import json
import scipy
import numpy as np
# import rasterio
import pandas as pd
import gdal
import h5py
from os import listdir
import os
import sys
import random
RGB_TRAIN_PATH = os.path.abspath(sys.argv[1])
MASK_TRAIN_PATH = os.path.abspath(sys.argv[2])
csv_depth = os.path.abspath(sys.argv[3])
parent = os.path.dirname(RGB_TRAIN_PATH)
# new processing not ban cut, not x mean
num_chanel = 1
INPUT_SIZE = 512
def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.png"):
        list_id.append(file[:-4])
        # print(file[:-4])
    return list_id

def cal_bancut_3band(image_list):
    band_values = {k: [] for k in range(4)}
    for im_name in image_list:
        image_path =  os.path.join(RGB_TRAIN_PATH,im_name + ".tif")
        print(image_path)
        dataset = gdal.Open(image_path)
        data = dataset.ReadAsArray()
        for i_chan in range(4):
            values_ = data[i_chan].ravel().tolist()
            values_ = np.array(
                [v for v in values_ if v != 0]
            )  # Remove sensored mask
            band_values[i_chan].append(values_)
    band_cut_th = {k: dict(max=0, min=0) for k in range(4)}
    for i_chan in range(4):
        band_values[i_chan] = np.concatenate(
            band_values[i_chan]).ravel()
        band_cut_th[i_chan]['max'] = scipy.percentile(
            band_values[i_chan], 98)
        band_cut_th[i_chan]['min'] = scipy.percentile(
            band_values[i_chan], 2)
    print(band_cut_th)
    # np.save('band_cut',band_cut_th)
    # with open('bandcut.txt', 'w') as outfile:  
    #     json.dump(band_cut_th, outfile)
    return band_cut_th

# def generate_image_list(fixed_for_test=[]):
#     img_names = [f[:-9] for f in listdir(RGB_TRAIN_PATH) if 'mask' in f]
#     prefixes = np.unique(np.array([f[0:6] for f in img_names]))
#     splits = {}
#     train_img = []
#     test_img = []
#     split_ratio = 0.8

#     if len(fixed_for_test) == 0:
#         for pre in prefixes:
#             splits[pre] = np.array([f for f in img_names if pre in f])
#             np.random.shuffle(splits[pre])
#             count = splits[pre].shape[0]
#             for train in splits[pre][:round(count*split_ratio)]:
#                 train_img.append(train)
#             for test in splits[pre][round(count*split_ratio):]:
#                 test_img.append(test)
#     else:
#          for pre in prefixes:
#             splits[pre] = np.array([f for f in img_names if pre in f])
#             if pre not in fixed_for_test:
#                 for train in splits[pre]:
#                     train_img.append(train)
#             else:
#                 print(fixed_for_test[0])
#                 for test in splits[pre]:
#                     test_img.append(test)

#     np.savetxt('train_img.txt',np.array(train_img), newline="\r\n", fmt="%s")
#     np.savetxt('test_img.txt',np.array(test_img), newline="\r\n", fmt="%s")
#     return train_img

def get_x(image_list,max_depth,min_depth,df_depths):
    BAND_CUT_RGB = {
        0:{
            "max":203,
            "min":0
        },
        1:{
            "max":203,
            "min":0
        },
        2:{
            "max":203,
            "min":0
        },
        3:{
        "max":203,
        "min":0
        }
    }
    X_val = []
    for im_name in image_list:
        fn = os.path.join(RGB_TRAIN_PATH, im_name + ".png")
        print(fn)
        # with rasterio.open(fn,'r') as f:
        #     values = f.read().astype(np.uint8)
        #     for chan_i in range(4):
        #         values[chan_i] = np.clip(values[chan_i], 0, 255)
        #     X_val.append(values)
        dataset = gdal.Open(fn)
        values = dataset.ReadAsArray()
        for chan_i in range(num_chanel):
            values[chan_i] = np.clip(values[chan_i], BAND_CUT_RGB[chan_i]["min"], BAND_CUT_RGB[chan_i]["max"])/(BAND_CUT_RGB[chan_i]["max"]-BAND_CUT_RGB[chan_i]["min"])*255
        depth = df_depths.loc[im_name,'z']
        depth_ = np.clip(depth, min_depth, max_depth)/(max_depth-min_depth)*255
        depth_matrix = np.zeros((INPUT_SIZE,INPUT_SIZE), dtype=np.uint8) + depth_
        result = []
        for chan_i in range(num_chanel):
            result.append(values[chan_i])
        del values
        result.append(depth_matrix)
        result = np.array(result)
        X_val.append(result.astype(np.uint8))
    return np.array(X_val)

def get_y(image_list):
    Y_val = []
    for im_name in image_list:
        fn = os.path.join(MASK_TRAIN_PATH,im_name +".png") 
        print(fn)
        # print(fn)
        # with rasterio.open(fn,'r') as f:
        #     values = f.read().astype(np.float32)            
        #     values = (values > 0.5).astype(np.uint8)
        #     Y_val.append(values)
        dataset = gdal.Open(fn)
        data = dataset.ReadAsArray()
        values = data.astype(np.float32)           
        values = (values > 0.5).astype(np.uint8)
        Y_val.append(values) 
    Y_val = np.array(Y_val).reshape((-1,1,INPUT_SIZE,INPUT_SIZE))
    return Y_val
def get_percentile_depth(csv_depth):    
    df_depths = pd.read_csv(csv_depth, index_col='id')
    data = df_depths['z']
    value = [i for i in data]
    values = np.array(value)
    max_val = scipy.percentile(values, 98)
    min_val = scipy.percentile(values, 2)
    return max_val,min_val,df_depths
def main():
    # image_list = generate_image_list(['SX9293'])
    image_list = create_list_id(RGB_TRAIN_PATH)
    np.random.shuffle(image_list)
    count = len(image_list)
    cut_idx = int(round(count*0.7))
    print(cut_idx)
    train_list = image_list[0:cut_idx]
    max_val,min_val,df_depths = get_percentile_depth(csv_depth)
    # val_list = image_list[cut_idx:count]
    val_list = [id_image for id_image in image_list if id_image not in train_list]
    # print(val_list)
    # band_cut = cal_bancut_3band(train_list)
    # print(band_cut)
    x_train = get_x(train_list,max_val,min_val,df_depths)
    x_val = get_x(val_list,max_val,min_val,df_depths)
    y_train = get_y(train_list)
    y_val = get_y(val_list)

    # x_mean = x_train.mean(axis = 0)

    # x_train = x_train - x_mean
    # x_val = x_val - x_mean
    
    # with open(os.path.join(parent,'band_cut2.txt'), 'w') as outfile:
    #     json.dump(band_cut, outfile)
    # np.save('xmean', x_mean)
    # np.save('xtrain', x_train)
    # np.save('ytrain', y_train)
    # np.save('xval', x_val)
    # np.save('yval', y_val)

    
    # f1 = h5py.File(os.path.join(parent,'x_mean2.hdf5'),'w')
    # f1.create_dataset('x_mean', data=x_mean)
    f2 = h5py.File(os.path.join(parent,'x_train_salt.hdf5'),'w')
    f2.create_dataset('x_train', data=x_train)
    f3 = h5py.File(os.path.join(parent,'y_train_salt.hdf5'),'w')
    f3.create_dataset('y_train', data=y_train)
    f4 = h5py.File(os.path.join(parent,'x_val_salt.hdf5'),'w')
    f4.create_dataset('x_val', data=x_val)
    f5 = h5py.File(os.path.join(parent,'y_val_salt.hdf5'),'w')
    f5.create_dataset('y_val', data=y_val)
    # with hdf.File('test.hdf5', 'r') as f:
    #     my_data = f['grid'].value
        
    # # with h5py.File('test.hdf5', 'r') as f:
    # #     my_data = f['grid'].value

if __name__ == '__main__':
    # rename()
    main()