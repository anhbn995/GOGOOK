# -*- coding: utf-8 -*-
"""
@author: ducanh
"""
import rasterio
import numpy as np
import tensorflow as tf
import os
import gdal
import cv2
from tqdm import *


FN_MODEL=r"/home/skm/SKM/WORK/ALL_CODE/Model/20220828_175708_val_weights_last_model_save_sumarry.h5"
INPUT_SIZE = 256

def predict(img_url, threshold = 0.5):
    csiz = INPUT_SIZE - 100
    model_ = tf.keras.models.load_model(FN_MODEL)
    # model_.load_weights(FN_MODEL)
    ds = gdal.Open(img_url)
    val = ds.ReadAsArray()[0:3].astype(np.uint8)
    h,w = val.shape[1:3]    
    p = int((INPUT_SIZE - csiz)/2)
    pdoi = []
    ci = []
    nw = w + 2*p
    nh = h + 2*p
    cw = list(range(p, nw - p, csiz))
    ch = list(range(p, nh - p, csiz))

    lh = []
    lw = []
    for i in ch:
        if i < nh - p - csiz:
            lh.append(i)
    lh.append(nh-csiz-p)

    for i in cw:
        if i < nw - csiz - p:
            lw.append(i)
    lw.append(nw-csiz-p)

    img_coo = []
    for i in lw:
        for j in lh:
            img_coo.append([i, j])
    
    for i in range(3):
        band = np.pad(val[i], p, mode='reflect')
        pdoi.append(band)

    val = np.array(pdoi).swapaxes(0,1).swapaxes(1,2)
    del pdoi

    def get_im_by_coord(org_im, s_x, s_y):
        sx = s_x-p
        ex = s_x+csiz+p
        sy = s_y - p
        dey = s_y+csiz+p
        result=[]
        img = org_im[sy:dey, sx:ex]
        img = img.swapaxes(2,1).swapaxes(1,0)
        for chan_i in range(3):
            result.append(cv2.resize(img[chan_i],(INPUT_SIZE, INPUT_SIZE), interpolation = cv2.INTER_CUBIC))
        return np.array(result).swapaxes(0,1).swapaxes(1,2)

    for i in range(len(img_coo)):
        im = get_im_by_coord(val, img_coo[i][0], img_coo[i][1])
        ci.append(im)

    a = list(range(0, len(ci), 1))

    if a[len(a)-1] != len(ci):
        a[len(a)-1] = len(ci)

    y_p = []
    for i in range(len(a)-1):
        x_b = []
        x_b = np.array(ci[a[i]:a[i+1]]).astype(np.float32)/255.0
        y_b = model_.predict(x_b)
        y_p.extend(y_b)
    bm = np.zeros((h, w)).astype(np.float16)
    for i in range(len(ci)):
        t_ma = y_p[i].reshape((INPUT_SIZE,INPUT_SIZE))
        t_ma = (t_ma>0.5).astype(np.uint8)
        t_ma = (cv2.resize(t_ma,(INPUT_SIZE, INPUT_SIZE), interpolation = cv2.INTER_CUBIC)>0.5).astype(np.uint8)
        s_x = img_coo[i][1]
        s_y = img_coo[i][0]
        bm[s_x-p:s_x-p+csiz, s_y-p:s_y -
                    p+csiz] = t_ma[p:p+csiz, p:p+csiz]

    del ci
    bm = (bm > 0.5).astype(np.uint8)
    return bm*255


def predict_all(base_path,outputFileName):

    mask_base = predict(base_path)
    with rasterio.open(base_path) as src:
        transform1 = src.transform
        w,h = src.width,src.height
        crs = src.crs
    new_dataset = rasterio.open(outputFileName, 'w', driver='GTiff',
                                height = h, width = w,
                                count=1, dtype="uint8",
                                crs=crs,
                                transform=transform1,
                                compress='lzw',
                                nodata=0)
    new_dataset.write(mask_base,1)
    new_dataset.close()

    

if __name__ == '__main__':

    in_file = r"/home/skm/SKM/AAA/Ver_them_uav/Img_UAV_test/img_resize/CuaDai2_uav.tif"
    out_file = r"/home/skm/SKM/AAA/Ver_them_uav/Img_UAV_test/aaaaaaaaaa/CuaDai2_uav.tif"
    predict_all(in_file, out_file)