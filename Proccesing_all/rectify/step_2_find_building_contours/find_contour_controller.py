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
from step_2_find_building_contours.kmean_image import kmean_tiles_cluster_color,true_image_cluster,kmean_tgi
from step_2_find_building_contours.kmean_dem_band import hight_cluster,kmean_tiles_cluster_hight
core_of_computer = multiprocessing.cpu_count()
HIGHR_VALUE = 3.0
LOOP_MORPHOLOGY = 100
MIN_BUILDING = 50.0
ANPHA_TGI = 2/11
TGI_RANGE = 7.0

def row_mask_building_cal(i ,dataox, data_mask, th4, w):
    for j in range(w):
        if data_mask[i,j] > 0:
            dataox[i,j]=255
            if th4[i,j]>0:
                dataox[i,j] = 0
    return dataox[i]
def mask_building_cal(data_mask,th4):
    data_mask = data_mask.copy()
    h,w = data_mask.shape
    liss2 = range(0,h)
    dataox = np.zeros(data_mask.shape, dtype=np.uint8)
    p2 = Pool(processes=core_of_computer)
    result = p2.map(partial(row_mask_building_cal,dataox = dataox, data_mask = data_mask, th4 = th4, w=w), liss2)
    p2.close()
    p2.join()
    data_result = np.asarray(result, dtype=np.uint8)
    return data_result

def load_data_CNN(input_data):
    mask,data3,dataset3 = input_data
    img = np.array(data3).swapaxes(0,1).swapaxes(1,2)
    return (mask,img,dataset3)

def load_data_DSM(input_data):
    #Tacch input_data thang data1, data2, data3
    data1, data2, data3, dataset3 = input_data
#    data3 = dataset3.ReadAsArray()
    img = np.array(data3).swapaxes(0,1).swapaxes(1,2) 
    # xu ly tren do cao
    data = data1 - data2
    # plt.imshow(data)
    th3 = kmean_tiles_cluster_hight(data)
    # plt.figure()
    # plt.imshow(th3)
    th2 = kmean_tiles_cluster_color(img,256)
    # plt.figure()
    # plt.imshow(th2)
    th4 = kmean_tgi(data3,ANPHA_TGI)
    # plt.figure()
    # plt.imshow(th4)
    th5 = kmean_tiles_cluster_color(img,512)
    data_color_mask = cv2.bitwise_or(th2,th5)
    # plt.figure()
    # plt.imshow(data_color_mask)
    # h,w = data_color_mask.shape
    data_mask = cv2.bitwise_and(th3,data_color_mask)
    # plt.figure()
    # plt.imshow(data_mask)
    mask = mask_building_cal(data_mask,th4)
    
    return (mask,img,dataset3)

def find_contour_main(input_data, problem_type):
    if problem_type == 'DSM':
        narr,img,dataset3 = load_data_DSM(input_data)
    elif problem_type == 'CNN':
        narr,img,dataset3 = load_data_CNN(input_data)
    
    # plt.figure()
    # plt.imshow(dataox)
    contours_pre, hierarchy_pre = cv2.findContours(narr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(narr.shape, dtype="uint8")
    for cnt in contours_pre:
        if cv2.contourArea(cnt)>MIN_BUILDING:
            ignore_mask_color = 255
            cnx = np.reshape(cnt, (1,len(cnt),2))
            cv2.fillPoly(mask, cnx, ignore_mask_color)
    dataox = mask
    # plt.figure()
    # plt.imshow(dataox)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    closing = cv2.morphologyEx(dataox, cv2.MORPH_CLOSE, kernel)
    for i in range(LOOP_MORPHOLOGY):
        closing= cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
        # closing= cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel2)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    for i in range(LOOP_MORPHOLOGY):
        opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
        # opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel2)
    # plt.figure()
    # plt.imshow(opening)

    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours2=[]
    for contour in contours:
       area = cv2.contourArea(contour) 
       if area > MIN_BUILDING:
           contours2.append(contour)

    return(img,contours2,dataset3)