#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 4 19:42:36 2021

@author: ducanh
"""

import os
import glob
from osgeo import gdal, osr, ogr
import rasterio
import numpy as np
import shutil

def get_min_max_image(file_path):
    ds = gdal.Open(file_path,  gdal.GA_ReadOnly)
    numband =  ds.RasterCount
    dict_band_min_max = {1:0}
    for i in range(4):
        print(dict_band_min_max)
        band = ds.GetRasterBand(i + 1)
        min_train, max_train, _, _ = band.GetStatistics(True, True)
        dict_band_min_max.update({ i+1 : {"min": min_train, "max":max_train}})
    return dict_band_min_max, numband


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
                        ) as dst:
                            dst.write(data)


def create_img_01(img_path, out_path):
    # get min max tat ca cac band va so band
    dict_min_max_full, numband = get_min_max_image(img_path)
    src = rasterio.open(img_path)
    img_float_01 = np.empty((numband, src.height, src.width))
    for i in range(numband):
        band = src.read(i+1)
        min_tmp = dict_min_max_full[i+1]['min']
        max_tmp = dict_min_max_full[i+1]['max']
        band = np.interp(band, (min_tmp, max_tmp), (0, 1))
        img_float_01[i] = band
    write_image(img_float_01, src.height, src.width, numband, src.crs, src.transform, out_path)


def main_gendata_old(list_path_file, tmp_dir):
    out_dir = os.path.join(tmp_dir, "data_genorator_01")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # print("list_path_file", list_path_file)
    for path_tmp in list_path_file:
        name_file = os.path.basename(path_tmp)
        out_file_tmp = os.path.join(out_dir, name_file)
        if os.path.exists(out_file_tmp):
            pass
        else:
            create_img_01(path_tmp, out_file_tmp)
    return out_dir

def main_gendata(list_path_file, tmp_dir):
    out_dir = os.path.join(tmp_dir, "data_genorator_01")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # print("list_path_file", list_path_file)
    for path_tmp in list_path_file:
        name_file = os.path.basename(path_tmp)
        out_file_tmp = os.path.join(out_dir, name_file)
        if os.path.exists(out_file_tmp):
            pass
        else:
            # create_img_01(path_tmp, out_file_tmp)
            shutil.copy2(path_tmp, out_file_tmp)
    return out_dir


