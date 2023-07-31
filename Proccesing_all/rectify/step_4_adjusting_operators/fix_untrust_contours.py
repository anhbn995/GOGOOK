#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:08:44 2018

@author: tranthang
"""

import numpy as np
import cv2
from step_4_adjusting_operators.find_main_axis import find_axis_by_bounding_rect,choise_vector_axis_unbound_untrust,find_anpha_axis_list_point
from lib.convert_datatype import contour_to_list_array,list_array_to_contour
from step_4_adjusting_operators.extract_contour_by_vector import extract_step_one_untrust
from step_4_adjusting_operators.find_main_axis import find_axis_by_min_degree_error
from lib.geometry_calculating import vector_of_degrees
import multiprocessing 
core_of_computer = multiprocessing.cpu_count()
from multiprocessing.pool import Pool
from functools import partial
# from shapely.geometry import *
# Hàm nắn chỉnh các untrust contours bằng bounding rect
# Đầu vào: danh sách các untrust contours
# Đầu ra: danh sách các new polygon (mỗi polygon là một rectangle)
def fix_untrust_contours_by_bounding_rect(list_untrust_contours,list_untrust_vectors,list_untrust_translative):
    print('fix_untrust_contours_by_bounding_rect')
    list_vector,list_box = find_axis_by_bounding_rect(list_untrust_contours)
    list_new_polygon = []
    list_unbound_untrust = []
    list_untrust_unbound_vector = []
    list_bound_untrust_translative = []
    list_unbound_untrust_translative = []
    for i in range(len(list_untrust_contours)):
        box_area = cv2.contourArea(list_box[i])
        cnt_area = cv2.contourArea(list_untrust_contours[i])
        offset_area = abs(box_area - cnt_area)
        quotient_area = cnt_area/box_area
        # print(offset_area)
        if quotient_area >= 0.8:
            list_new_polygon.append(list_box[i])
            list_bound_untrust_translative.append(list_untrust_translative[i])
        elif quotient_area >= 0.5 and offset_area <= 2000.0:
            list_new_polygon.append(list_box[i])
            list_bound_untrust_translative.append(list_untrust_translative[i])
        elif cnt_area <= 1000:
            list_new_polygon.append(list_box[i])
            list_bound_untrust_translative.append(list_untrust_translative[i])
        else:
            list_unbound_untrust.append(list_untrust_contours[i])
            list_untrust_unbound_vector.append(list_untrust_vectors[i])
            list_unbound_untrust_translative.append(list_untrust_translative[i])
    return list_new_polygon, list_unbound_untrust, list_untrust_unbound_vector, list_bound_untrust_translative, list_unbound_untrust_translative

def fix_untrust_unbound(list_unbound_untrust,list_untrust_unbound_vector):
    print('fix_untrust_unbound')
    list_vector_rec,list_box = find_axis_by_bounding_rect(list_unbound_untrust)
    # list_vector = choise_vector_axis_unbound_untrust(list_unbound_untrust,list_vector_rec,list_untrust_unbound_vector)
    list_i = list(range(len(list_unbound_untrust)))
    p_strange = Pool(processes=core_of_computer)
    result = p_strange.map(partial(fix_untrust_unbound_pool,list_unbound_untrust=list_unbound_untrust,list_vector_rec=list_vector_rec,list_untrust_unbound_vector=list_untrust_unbound_vector), list_i)
    p_strange.close()
    p_strange.join()
    fixed_unbound_untrust = result
   
    return fixed_unbound_untrust
def fix_untrust_unbound_pool(i,list_unbound_untrust,list_vector_rec,list_untrust_unbound_vector):
    try:
        list_array = contour_to_list_array(list_unbound_untrust[i])
        list_array_rs1 = extract_step_one_untrust(list_array,list_vector_rec[i])
        list_array_rs2 = extract_step_one_untrust(list_array,list_untrust_unbound_vector[i])
        dist1 = 0
        for point1 in list_array_rs1:
            dist = abs(cv2.pointPolygonTest(list_unbound_untrust[i],tuple(point1),True))
            dist1 = dist1 + dist
        dist2 = 0
        for point2 in list_array_rs2:
            distx = abs(cv2.pointPolygonTest(list_unbound_untrust[i],tuple(point2),True))
            dist2 = dist2 + distx
        # print(dist1,dist2)
        if dist1<dist2:
            list_array_rs = list_array_rs1
        else:
            list_array_rs = list_array_rs2
#        a1,ret1 = find_anpha_axis_list_point(list_array_rs1)
#        a2,ret2 = find_anpha_axis_list_point(list_array_rs2)
##        print(ret1,ret2)
#        area = cv2.contourArea(list_unbound_untrust[i])
#        if area>2000:
#            list_array_rs = list_array_rs1
##        if  ret1 <= 0.3 and ret2>0.25:
##            list_array_rs = list_array_rs1
##        elif ret2<0.4:
##            list_array_rs = list_array_rs2
#        else:
#            list_array_rs = list_array_rs2
#        epsilon = 0.02*cv2.arcLength(list_unbound_untrust[i],True)
#        approx = cv2.approxPolyDP(list_unbound_untrust[i],epsilon,True)
##        approx = cv2.convexHull(list_unbound_untrust[i])
#        approx = approx.reshape(len(approx),1,2)
#        phi,error_min = find_anpha_axis_list_point(list_array)
#        vector_dgr = vector_of_degrees(phi)
#        list_array_rs1 = extract_step_one_untrust(list_array,vector_dgr)
#        contour_rs = list_array_to_contour(list_array_rs1)
#        contour_rs = approx
        contour_rs = list_array_to_contour(list_array_rs)
    except Exception:
        contour_rs = list_unbound_untrust[i]
    return contour_rs
