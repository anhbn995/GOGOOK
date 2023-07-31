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


def mosaic1(dir_path, list_img_name, out_path):
    """ 
        Trong list danh sach thi "o dau list se la anh tren cung"
    """
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

def mosaic3(list_fp, out_path):
    src_files_to_mosaic = []
    for fp in list_fp:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)
    mosaic, out_trans = merge(src_files_to_mosaic)
    write_image(mosaic, mosaic.shape[1], mosaic.shape[2], mosaic.shape[0], src.crs, out_trans, out_path)

# dir_path = r"/home/skm/SKM16/Work/Npark_planet/Z_Tat_ca_anh_roi_rac/img_origin_remove_cloud/rm_t4/t4"
# list_names = get_list_name_file(dir_path) 
# list_names = [
#                 "20220422_024818_68_2262_3B_AnalyticMS_SR_8b_clip.tif",
#                 "20220423_030623_72_247d_3B_AnalyticMS_SR_8b_clip.tif",
#                 "20220404_032520_95_2414_3B_AnalyticMS_SR_8b_clip.tif",
#                 "20220404_030352_98_2480_3B_AnalyticMS_SR_8b_clip.tif",
#                 "20220404_023232_89_2442_3B_AnalyticMS_SR_8b_clip.tif"

#             ]
# out_path =r"/home/skm/SKM16/Work/Npark_planet/Z_Tat_ca_anh_roi_rac/img_origin_remove_cloud/rm_t4/T4_2022_percent.tif"
# mosaic(dir_path, list_names, out_path)
# print("done")
if __name__=="__main__":
    # dir_path = r"/home/skm/SKM16/Work/Npark_planet/A_OKE_4326/tmp"
    # list_names = get_list_name_file(dir_path) 
    # list_names = [
    #                 "T4_2022_4326.tif",
    #                 "T3_2022_4326.tif"

    #             ]
    # out_path =r"/home/skm/SKM16/Work/Npark_planet/A_OKE_4326/tmp/rs.tif"
    # mosaic(dir_path, list_names, out_path)
    # print("done")
    list_name = [   
                "01_July_Mosaic_P_2",
                "01_July_Mosaic_P_3",
                "01_July_Mosaic_P_4",
                "01_July_Mosaic_P_5",
                "01_July_Mosaic_P_6",
                "02_May_Mosaic_P_2",
                "03_July_Mosaic_P_2"
                ]
    from tqdm import tqdm
    for name in tqdm(list_name):
        dir_path = "/home/skm/SKM16/Work/SonalPanel_ThaiLand/1Ver2_lable2/izmages_8bit_perizmage/images_per95/tmp_forpredict_big/CaraNet-best-solar-882022Sep06-14h57m48scrop_predict/" + name
        list_fp = glob.glob(os.path.join(dir_path, "*.tif"))
        out_path = dir_path+ ".tif"
        mosaic3(list_fp, out_path)

