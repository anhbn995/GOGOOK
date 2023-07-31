# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 09:53:50 2017

@author: lamng
"""
import cv2
import numpy as np
from multiprocessing.pool import Pool
from functools import partial
import gdal
import multiprocessing
import os
import matplotlib.pyplot as plt
PATH_MAIN = os.getcwd()
core_of_computer = multiprocessing.cpu_count()
HIGHR_VALUE = 3.0
LOOP_MORPHOLOGY = 100
MIN_BUILDING = 200.0
def row_hightfilter_cal(i,w,data):
    for j in range(0,w):
        u = data[i,j]
        t = u<= HIGHR_VALUE

        if t:
            data[i,j]=0
    return data[i]

def highfilter(data):
    h,w = data.shape
    liss = range(0,h)
    p = Pool(processes=core_of_computer)
    result = p.map(partial(row_hightfilter_cal, w=w, data = data), liss)
    p.close()
    p.join()
    data_rs = np.asarray(result, dtype=np.float32)
    return data_rs

def hight_cluster(dem_band):
    data = dem_band
    x = data.max()
    scale = 255/x
    h,w = data.shape
    data = highfilter(dem_band)
    # plt.imshow(data)
    datax = data*scale
    datax = datax.astype(np.uint8)
    Z1 = datax.reshape((-1,1))
    Z1 = np.float32(Z1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    K1 = 3
    ret,label,center=cv2.kmeans(Z1,K1,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    # print(center)
    res = center[label.flatten()]
    res1 = res.reshape((datax.shape))
    # plt.imshow(res1)
    # res2 = highfilter(res1,scale)
    ret3,th3 = cv2.threshold(res1,HIGHR_VALUE*scale, 255, 0)
    # plt.figure(1)
    # plt.imshow(res1)
    # plt.figure(2)
    # plt.imshow(th3)
    # plt.figure(3)
    # plt.imshow(dem_band)
    return th3

def kmean_tiles_cluster_hight(data):
    (h,w) = data.shape
    data2 = data.copy()
    data2 = highfilter(data2)
    th_result_2 = np.zeros((h,w), dtype=np.uint8)
    list_hight = list(range(0,h,256))
    list_weight = list(range(0,w,256))
    if list_hight[len(list_hight)-1] < h:
        list_hight.append(h)
    if list_weight[len(list_weight)-1] < w:
        list_weight.append(w)

    for i in range(len(list_hight)-1):
        hight_tiles_up = list_hight[i]
        hight_tiles_down = list_hight[i+1]
        for j in range(len(list_weight)-1):
            weight_tiles_up = list_weight[j]
            weight_tiles_down = list_weight[j+1]
            data_crop = data2[hight_tiles_up:hight_tiles_down, weight_tiles_up:weight_tiles_down]
            x = data_crop.max()
            scale = 255/x
            print('hight')
            # plt.imshow(data)
            data_cropx = data_crop*scale
            data_cropx = data_cropx.astype(np.uint8)
            Z1 = data_cropx.reshape((-1,1))
            Z1 = np.float32(Z1)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
            K1 = 5
            ret,label,center=cv2.kmeans(Z1,K1,None,criteria,15,cv2.KMEANS_RANDOM_CENTERS)
            # Now convert back into uint8, and make original image
            center = np.uint8(center)
            # print(center)
            res = center[label.flatten()]
            res1 = res.reshape((data_cropx.shape))
            # plt.imshow(res1)
            # res2 = highfilter(res1,scale)
            ret3,th3 = cv2.threshold(res1,HIGHR_VALUE*scale, 255, 0)
            th_result_2[hight_tiles_up:hight_tiles_down, weight_tiles_up:weight_tiles_down] = th3
    return th_result_2
