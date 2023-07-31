#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 4 23:17:10 2021

@author: ducanh
"""
import os
import glob
import numpy as np
import rasterio
from rasterio.merge import merge
import gdalnumeric


def get_list_name_file(path_folder, name_file = '*.tif'):
    """
        Get all file path with file type is name_file.
    """
    list_img_dir = []
    for file_ in glob.glob(os.path.join(path_folder, name_file)):
        head, tail = os.path.split(file_)
        list_img_dir.append(tail)
    return list_img_dir


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


def mosaic(dir_path, list_img_name, out_path):
    src_files_to_mosaic = []
    for name_f in list_img_name:
        fp = os.path.join(dir_path, name_f)
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
    mosaic, out_trans = merge(src_files_to_mosaic)
    write_image(mosaic, mosaic.shape[1], mosaic.shape[2], mosaic.shape[0], src.crs, out_trans, out_path)


def mosaic2(list_fp):
    src_files_to_mosaic = []
    for fp in list_fp:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
    mosaic, _ = merge(src_files_to_mosaic)
    return mosaic


dir_path = r"/home/skm/SKM16/Work/Npark_planet/Z_Tat_ca_anh_roi_rac/img_origin_remove_cloud/rm_t4/t4"
list_names = get_list_name_file(dir_path) 
list_names = [
                "20220422_024818_68_2262_3B_AnalyticMS_SR_8b_clip.tif",
                "20220423_030623_72_247d_3B_AnalyticMS_SR_8b_clip.tif",
                "20220404_032520_95_2414_3B_AnalyticMS_SR_8b_clip.tif",
                "20220404_030352_98_2480_3B_AnalyticMS_SR_8b_clip.tif",
                "20220404_023232_89_2442_3B_AnalyticMS_SR_8b_clip.tif"

            ]
out_path =r"/home/skm/SKM16/Work/Npark_planet/Z_Tat_ca_anh_roi_rac/img_origin_remove_cloud/rm_t4/T4_2022_percent.tif"
mosaic(dir_path, list_names, out_path)
print("done")