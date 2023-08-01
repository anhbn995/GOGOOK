

import math
import glob
import json
import scipy
import numpy as np
import rasterio
import pandas as pd
import gdal
import h5py
from os import listdir
import os
import sys
import random
import cv2
import skimage.transform
INPUT_SIZE = 256
def create_list_id(path):
    list_id = []
    os.chdir(path)
    types = ('*.png', '*.tif')
    for files in types:
        for file in glob.glob(files):
            list_id.append(file)        
    return list_id

def split_black_image(list_id, mask_dir, thres):
    negative_fn = []
    for fn in list_id:
        dataset = gdal.Open(os.path.join(mask_dir,fn))
        values = dataset.ReadAsArray()        
        h,w = values.shape[0:2]
        # print(values.shape)
        if np.count_nonzero((values==255).astype(np.uint8)) <= thres*h*w:
            negative_fn.append(fn)
    positive_fn = [id_image for id_image in list_id if id_image not in negative_fn]
    count = len(negative_fn)
    print("Black mask count: " + str(count) + "/" +str(len(list_id)))
    return positive_fn, negative_fn

def rotate_angle(org_matrix, angle):
    m1 = np.rot90(org_matrix)
    m2 = np.rot90(m1)
    m3 = np.rot90(m2)
    if angle == "90":
        return m1
    elif angle == "180":
        return m2
    elif angle == "270":
        return m3

def data_gen(positive_fn, negative_fn, image_path, mask_path, batch_size, scale_neg,num_chanel,augment=None):
    
    # print(list_id_gen)
    while True:
        if len(negative_fn) >0:
            index_neg=np.random.choice(len(negative_fn),int(len(positive_fn)*scale_neg))
            # print(index_neg)
            list_neg_train = [negative_fn[idx] for idx in index_neg]
        else :
            list_neg_train=[]
        # print(positive_fn)
        list_id_gen = [idx for idx in positive_fn]
        list_id_gen.extend(list_neg_train)
        np.random.shuffle(list_id_gen)
        indx= np.random.choice(len(list_id_gen),batch_size)                         
        x = []
        y = []
        gen_id = []
        for idx in indx: 
            im_name = list_id_gen[idx]         
            image = get_x(im_name,image_path, num_chanel)
            mask = get_y(im_name,mask_path)
            if augment:
                # logging.warning("'augment' is depricated. Use 'augmentation' instead.")
                if random.randint(0, 1):
                    aug_index1 = random.randint(0, 2)
                    if aug_index1 == 0:
                        image = np.fliplr(image)
                        mask = np.fliplr(mask)
                    elif aug_index1 == 1:
                        image = np.flipud(image)
                        mask = np.flipud(mask)
                    aug_index2 = random.randint(0, 3)
                    if aug_index2 == 0:
                        image = rotate_angle(image, "90")
                        mask = rotate_angle(mask, "90")
                    elif aug_index2 == 1:
                        image = rotate_angle(image, "180")
                        mask = rotate_angle(mask, "180")
                    elif aug_index2 == 2:
                        image = rotate_angle(image, "270")
                        mask = rotate_angle(mask, "270")
                    
            # print(image.shape)
            # print(image.shape)
            gen_id.append(im_name)
            x.append(image)
            y.append(mask)
            # print(np.array(x).shape)
            # print(np.array(y).reshape((-1,1,INPUT_SIZE,INPUT_SIZE)).shape)
        # print(' Images id ' + str(gen_id))            
        yield np.array(x), np.array(y)

def get_x(im_name,img_path,num_chanel):

    BAND_CUT_RGB = {
        0:{
            "max":255,
            "min":0
        },
        1:{
            "max":255,
            "min":0
        },
        2:{
            "max":255,
            "min":0
        },
        3:{
            "max":255,
            "min":0
        }
    }
    fn = os.path.join(img_path, im_name)
    # print(fn)
    # with rasterio.open(fn,'r') as f:
    #     values = f.read().astype(np.uint8)
    #     for chan_i in range(4):
    #         values[chan_i] = np.clip(values[chan_i], 0, 255)
    #     X_val.append(values)
    dataset = gdal.Open(fn)
    values = dataset.ReadAsArray()
    # for chan_i in range(num_chanel):
    #     values[chan_i] = np.clip(values[chan_i], BAND_CUT_RGB[chan_i]["min"], BAND_CUT_RGB[chan_i]["max"])/(BAND_CUT_RGB[chan_i]["max"]-BAND_CUT_RGB[chan_i]["min"])*255
    result = []
    for chan_i in range(num_chanel):
        result.append(cv2.resize(values[chan_i],(INPUT_SIZE, INPUT_SIZE), interpolation = cv2.INTER_CUBIC))
    del values
    result = np.array(result).astype(np.float32)
    # return result/255.0
    return result.swapaxes(0,1).swapaxes(1,2)/255.0

def get_y(im_name,mask_path):   
    y_file = os.path.join(mask_path, im_name)
    with rasterio.open(y_file, 'r') as f:
        values = f.read().astype(np.uint8)
    mask = np.array((values[0]==255).astype(np.uint8))
    mask2 = np.array((cv2.bitwise_not(values[0])==255).astype(np.uint8))
    mask = cv2.resize(mask,(INPUT_SIZE, INPUT_SIZE), interpolation = cv2.INTER_CUBIC)
    mask2 = cv2.resize(mask2,(INPUT_SIZE, INPUT_SIZE), interpolation = cv2.INTER_CUBIC)
    mask_all = np.stack((mask2,mask),axis=-1)
    return np.array(mask_all).astype(np.uint8)