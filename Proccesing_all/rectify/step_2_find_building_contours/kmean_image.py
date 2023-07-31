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


def kmean_tiles_cluster_color(img,size):
    (h,w,c) = img.shape
    img2 = img.copy()
    th_result = np.zeros((h,w), dtype=np.uint8)
    list_hight = list(range(0,h,size))
    list_weight = list(range(0,w,size))
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
            img_crop = img2[hight_tiles_up:hight_tiles_down, weight_tiles_up:weight_tiles_down]
            img_crop2 = cv2.GaussianBlur(img_crop,(5,5),0)
            Z = img_crop2.reshape((-1,3))
            check = Z.tolist().count([0,0,0])/float(len(Z))
            print("kmean")
            # convert to np.float32
            Z = np.float32(Z)

            # define criteria, number of clusters(K) and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
            if check >= 0.05:
                K = 3
            else:
                K = 2
            ret,label,center=cv2.kmeans(Z,K,None,criteria,15,cv2.KMEANS_RANDOM_CENTERS)

            # Now convert back into uint8, and make original image
            center = np.uint8(center)
            res = center[label.flatten()]
            res2 = res.reshape((img_crop.shape))
            # plt.imshow(res2)
            # max_gray = 0
            # tgi_min = float("+inf")
            # tgi_max = 0
            # for cent in center:
            #     tgi = cent[1] - 0.39*cent[0] - 0.61*cent[2]
            #     if tgi<tgi_min:
            #         tgi_min = tgi
            #     if tgi>tgi_max:
            #         tgi_max = tgi

            # if tgi_min > 4.0:
            #     th2 = np.zeros(img_crop.shape[0:2], dtype=np.uint8)
            # else:
            gray = cv2.cvtColor(res2, cv2.COLOR_RGB2GRAY)
            max_gray = gray.max()-1
            if max_gray<75.0:
                max_gray = 75.0
            ret2,th2 = cv2.threshold(gray,max_gray,255,0)
            th_result[hight_tiles_up:hight_tiles_down, weight_tiles_up:weight_tiles_down] = th2
    return th_result

def true_image_cluster(img):
    img2 = cv2.GaussianBlur(img,(5,5),0)
    Z = img2.reshape((-1,3))
    check = Z.tolist().count([0,0,0])/float(len(Z))
    print(check)
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    if check >= 0.05:
        K = 3
    else:
        K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    # plt.imshow(res2)
    gray = cv2.cvtColor(res2, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret2,th2 = cv2.threshold(gray2,gray.max()-5,255,0)
    # plt.imshow(th2)
    return th2
def kmean_tgi(data3,ANPHA_TGI):
    tgi = data3[1] - 0.39*data3[0] - 0.61*data3[2]
    (h,w) = tgi.shape
    th_result_tgi = np.zeros((h,w), dtype=np.uint8)
    # plt.figure()
    # plt.imshow(tgi)
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
            print('tgi')
            weight_tiles_up = list_weight[j]
            weight_tiles_down = list_weight[j+1]
            tgi_crop = tgi[hight_tiles_up:hight_tiles_down, weight_tiles_up:weight_tiles_down]
            Z1 = tgi_crop.reshape((-1,1))
            Z1 = np.float32(Z1)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.1)
            K1 = 2
            ret,label,center=cv2.kmeans(Z1,K1,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
            # Now convert back into uint8, and make original image
            # center = np.uint8(center)
            # print(center)
            res = center[label.flatten()]
            res1 = res.reshape((tgi_crop.shape))
            # scale_tgi = res1.max() - (res1.max()-res1.min())*0.1
            # if center.max() > 5.0:
            ret2,th4 = cv2.threshold(tgi_crop,5.0,255,0)
            # else:
            # th4 = np.zeros(tgi_crop.shape, dtype=np.uint8)
            th_result_tgi[hight_tiles_up:hight_tiles_down, weight_tiles_up:weight_tiles_down] = th4
    return th_result_tgi
