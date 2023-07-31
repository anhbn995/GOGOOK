#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 4 19:42:36 2021

@author: ducanh
"""

import os
import glob
import cv2
import gdal
import gdalnumeric
import rasterio
import numpy as np
from rasterio.merge import merge

def write_image(data, height, width, numband, crs, tr, out):
    """
        Export numpy array to image by rasterio.
    """
    with rasterio.open(
                        out,
                        'w',
                        driver='GTiff',
                        height=height,
                        width=width,
                        count=numband,
                        dtype=data.dtype,
                        crs=crs,
                        transform=tr,
                        nodata=0,
                        ) as dst:
                            dst.write(data)


def get_list_name_file(path_folder, name_file = '*.tif'):
    """
        Get all file path with file type is name_file.
    """
    list_img_dir = []
    for file_ in glob.glob(os.path.join(path_folder, name_file)):
        _, tail = os.path.split(file_)
        list_img_dir.append(tail)
    return list_img_dir

def sort_list_file_by_cloud(dir_cloud_nodata):
    list_fname = get_list_name_file(dir_cloud_nodata)
    dict_name = dict.fromkeys(list_fname)
    for fname in list_fname:
        fp = os.path.join(dir_cloud_nodata, fname)
        raster_file = gdalnumeric.LoadFile(fp)
        count = (raster_file[0]!=255).sum()
        dict_name[fname]=count
    dict_name_sort = sorted(dict_name.items(), key=lambda x: x[1], reverse=True)
    list_sort_name = list(dict(dict_name_sort).keys())
    return list_sort_name

def sort_list_file_by_date(list_fp_img_selected):
    list_sort_name = []
    for fp in list_fp_img_selected:
        name = os.path.basename(fp)
        list_sort_name.append(name)
    return list_sort_name

def sort_path(list_fp_img_selected, dir_cloud_nodata, sort_amount_of_clouds, first_image):
    if sort_amount_of_clouds:
        list_fn_sort = sort_list_file_by_cloud(dir_cloud_nodata)
    else:
        list_fn_sort = sort_list_file_by_date(list_fp_img_selected)

    if first_image:
        name = os.path.basename(first_image)
        list_fn_sort.remove(name)
        list_fn_sort.insert(0, name)
    return list_fn_sort


def get_index_cloud_for_4band(mask_cloud, numband, size_modify=10):
    """
        get anotation cloud
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size_modify,size_modify))
    mask_cloud = cv2.dilate(mask_cloud,kernel,iterations = 1)
    img_4band = np.empty((numband, mask_cloud.shape[0], mask_cloud.shape[1]))
    for i in range(numband):
        img_4band[i] = mask_cloud
    index_cloud = np.where(img_4band != 0)
    return index_cloud


def get_min_max_image(file_path):
    ds = gdal.Open(file_path,  gdal.GA_ReadOnly)
    numband =  ds.RasterCount
    dict_band_min_max = {1:0}
    for i in range(numband):
        band = ds.GetRasterBand(i + 1)
        min_train, max_train, _, _ = band.GetStatistics(True, True)
        dict_band_min_max.update({ i+1 : {"min": min_train, "max":max_train}})
    return dict_band_min_max


def create_img_01(raster_img, dict_min_max_full):
    numband, h, w = raster_img.shape
    img_float_01 = np.empty((numband, h, w))
    for i in range(numband):
        min_tmp = dict_min_max_full[i+1]['min']
        max_tmp = dict_min_max_full[i+1]['max']
        img_float_01[i] = np.interp(raster_img[i], (min_tmp, max_tmp), (0, 1))
    return img_float_01

def mosaic(dir_path, list_img_name, out_path):
    src_files_to_mosaic = []
    for name_f in list_img_name:
        fp = os.path.join(dir_path, name_f)
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
    mosaic_, out_trans = merge(src_files_to_mosaic)
    write_image(mosaic_, mosaic_.shape[1], mosaic_.shape[2], mosaic_.shape[0], src.crs, out_trans, out_path)


# dir_path = r"/home/skm/SKM_OLD/public/DA/2_GreenSpaceSing/Training_Stack_6band/Data_01/Sen1_origin"
# out_dir = r"/home/skm/SKM_OLD/public/DA/2_GreenSpaceSing/Training_Stack_6band/Data_01/Sen1_01"

# dir_path=r"/home/skm/SKM16/Work/Npark_planet/image_roi_rac/all_image_cloud"
# out_dir=r"/home/skm/SKM16/Work/Npark_planet/image_roi_rac/all_image_cloud_01"

# dir_path=r"/home/skm/SKM16/Work/Npark_planet/image_roi_rac/tmp/img2"
# out_dir=r"/home/skm/SKM16/Work/Npark_planet/image_roi_rac/tmp/img2_01"

# dir_path=r"/home/skm/SKM16/Work/Npark_planet/Z_Tat_ca_anh_roi_rac/img__origin/t4_add/a"
# out_dir=r"/home/skm/SKM16/Work/Npark_planet/Z_Tat_ca_anh_roi_rac/img__origin/t4_add/a_01"

dir_path=r"/home/skm/SKM16/Work/SonalPanel_ThaiLand/Data_origin/image/a"
out_dir=r"/home/skm/SKM16/Work/SonalPanel_ThaiLand/image_01"

list_name = get_list_name_file(dir_path)
for name in list_name:
    fp = os.path.join(dir_path, name)
    out_dir_t = os.path.join(out_dir, name[:-4])
    if not os.path.exists(out_dir_t):
        os.makedirs(out_dir_t)
    out = os.path.join(out_dir_t, name[:-4] + "_sen1.tif")
    dict_ = get_min_max_image(fp)
    src = rasterio.open(fp)
    raster_img = src.read()
    img_01 = create_img_01(raster_img, dict_)
    write_image(img_01, src.height, src.width, src.count, src.crs, src.transform, out)




