import numpy as np
from math import pi
import math
import cv2
import multiprocessing
from numpy.linalg import norm
from lib.geometry_calculating import vector_of_degrees,angle_vecto,remove_duple_item
from lib.convert_datatype import contour_to_list_array
from step_2_find_building_contours.find_contour_controller import find_contour_main
core_of_computer = multiprocessing.cpu_count()
from multiprocessing.pool import Pool
from functools import partial
def find_axis_by_bounding_rect(list_contours):
    list_vector = []
    list_box = []
    #draw_contour(list_contours[8], img, 'test')
    p_box = Pool(processes=core_of_computer)
    result = p_box.map(partial(find_axis_by_bounding_rect_pool), list_contours)
    p_box.close()
    p_box.join()
    for rs in result:
        list_vector.append(rs[0])
        list_box.append(rs[1])
    return list_vector,list_box
    
def find_axis_by_bounding_rect_pool(cnt):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = box.reshape(len(box),1,2)
    # box = np.int0(box)
    box_array = contour_to_list_array(box)
    vector = box_array[1] - box_array[0]
    return vector,box
    #####################################
def error_degree(v1,v2):
    angle = angle_vecto(v1,v2)
    # print(angle)
    if angle <= 45.0:
        angle_return = angle
    elif 45<angle <= 90.0:
        angle_return = 90.0 - angle
    elif 90.0<angle <= 135.0:
        angle_return = angle - 90.0
    elif 135.0<angle <= 180.0:
        angle_return = 180.0 - angle
    else:
        angle_return = 0
    return angle_return

def cal_list_vector(list_array):
    list_array1 = list(list_array)
    list_array1.append(list_array[0])
    list_vector = []
    for i in range(len(list_array1)-1):
        vector = list_array1[i+1]- list_array1[i]
        list_vector.append(vector)
    return list_vector

def cal_list_distance(list_array):
    list_array1 = list(list_array)
    list_array1.append(list_array[0])
    list_distance = []
    for i in range(len(list_array1)-1):
        vector = list_array1[i+1]- list_array1[i]
        distance = norm(vector)
        list_distance.append(distance)
    perimeter = sum(list_distance)
    return list_distance,perimeter

def find_anpha_axis_list_point(list_array):
    list_vector = cal_list_vector(list_array)
    list_distance,perimeter = cal_list_distance(list_array)
    # print(perimeter)
    error_min = float("+inf")
    for anpha in range(91):
        error_SL = 0.0
        vector_anpha = vector_of_degrees(anpha)
        for j in range(len(list_vector)):
            error_de = error_degree(vector_anpha,list_vector[j])
        #    if anpha ==62:
            #    print(list_distance[j]/(perimeter),error_de)
        #    print(list_distance[j]/perimeter)
            error_SL += error_de*list_distance[j]/(perimeter*45.0)
    #    print(error_SL)
        if error_SL < error_min:
            error_min = error_SL
            phi = anpha
    return phi,error_min
def calculating_error_by_vector_axis(contour,vector_axis):
    list_array = contour_to_list_array(contour)
    list_vector = cal_list_vector(list_array)
    list_distance,perimeter = cal_list_distance(list_array)
    error_SL = 0.0
    for j in range(len(list_vector)):
        error_de = error_degree(vector_axis,list_vector[j])
    #    if anpha ==62:
        #    print(list_distance[j]/(perimeter),error_de)
    #    print(list_distance[j]/perimeter)
        error_SL += error_de*list_distance[j]/(perimeter*45.0)
    return error_SL
def find_axis_pool(contour):
    try:
        list_array = contour_to_list_array(contour)
        list_array = remove_duple_item(list_array)
        phi,error_min = find_anpha_axis_list_point(list_array)
        vector_dgr = vector_of_degrees(phi)
    except Exception:
        vector_dgr = np.array([0,1])
        error_min = 0.0

    return vector_dgr,error_min

def find_axis_by_min_degree_error(list_cntstest):
    list_vector = []
    list_error = []
    p_axis = Pool(processes=core_of_computer)
    result = p_axis.map(partial(find_axis_pool), list_cntstest)
    p_axis.close()
    p_axis.join()
    for rs in result:
        list_vector.append(rs[0])
        list_error.append(rs[1])
    return list_vector,list_error

def choise_vector_axis_unbound_untrust(list_unbound_untrust,list_vector_rec,list_vector_untrust_unbound):
    list_vector_axis = []
    for i in range(len(list_unbound_untrust)):
        # print(calculating_error_by_vector_axis(list_unbound_untrust[i],list_vector_rec[i]))
        if calculating_error_by_vector_axis(list_unbound_untrust[i],list_vector_rec[i]) < 0.6:
            list_vector_axis.append(list_vector_rec[i])
        else:
            list_vector_axis.append(list_vector_untrust_unbound[i])
    return list_vector_axis