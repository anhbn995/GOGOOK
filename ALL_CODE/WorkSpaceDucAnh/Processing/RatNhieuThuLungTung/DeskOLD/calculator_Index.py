# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:34:02 2021

@author: SkyMap
"""
import os
import rasterio
import numpy as np

def get_name_file(folder_img_in):
    list_file = []
    for file in os.listdir(folder_img_in):
        if file.endswith(".tif"):
            # print(os.path.join(folder_img_in, file))
            # print(file)
            list_file.append(file)
    return list_file

def read_image(img_path):
    src = rasterio.open(img_path)
    tr = src.transform
    h = src.height
    w = src.width
    crs = src.crs
    return src, h, w, tr, crs

def write_image(data, height, width, numband, crs, tr, out):
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

"""AVI"""    
def cal_avi_index(src):
    B3 = src.read(3)
    B4 = src.read(4)
    X1 = (B4 + 1)**(1/3)
    X2 = (65536 - B3)**(1/3)
    X3 = (B4 - B3)**(1/3)
    return X1*X2*X3

"""AVI scale"""
def scale_avi(avi_np):
    avi_np *=100.0/avi_np.max()
    return avi_np

"""SI"""    
def cal_si_index(src):
    B1 = src.read(1)
    B2 = src.read(2)
    B3 = src.read(3)
    X1 = (65536 - B1)**(1/3)
    X2 = (65536 - B2)**(1/3)
    X3 = (65536 - B3)**(1/3)
    return X1*X2*X3

"""SSI"""
def cal_ssi_index(si_np):
    si_np *= 100.0/si_np.max() 
    return si_np

"""EVI"""
def cal_evi_index(src):
    nir = src.read(4)
    red = src.read(3)
    blue = src.read(1)
    L = 1
    C1 = 6
    C2 = 7.5
    return 2.5*(nir - red)/(L + nir + C1*red - C2*blue)


if __name__ == "__main__":
    """Chay nhieu anh"""    
    folder_path = r"C:\Users\SkyMap\Desktop\XongXoa"
    
    out_folder = os.path.join(folder_path, "result_index")
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
            
    list_file_name = get_name_file(folder_path) 
    for file_name in list_file_name:
        img_path = os.path.join(folder_path, file_name)
        src, h, w, tr, crs = read_image(img_path)
        
        # avi_np = cal_avi_index(src)
        # out_path_avi = os.path.join(out_folder, "avi_" + file_name)
        # write_image(np.array([avi_np]), h, w, 1, crs, tr, out_path_avi)
        
        # scale_avi_np = scale_avi(avi_np)
        # out_path_scale_avi = os.path.join(out_folder, "scale_avi_" + file_name)
        # write_image(np.array([scale_avi_np]), h, w, 1, crs, tr, out_path_scale_avi)
        # del avi_np, scale_avi_np
        
        # si_np = cal_si_index(src)
        # out_path_si = os.path.join(out_folder, "si_" + file_name)
        # write_image(np.array([si_np]), h, w, 1, crs, tr, out_path_si)  
        
        # ssi_np = cal_ssi_index(si_np)
        # out_path_ssi = os.path.join(out_folder, "ssi_" + file_name)
        # write_image(np.array([ssi_np]), h, w, 1, crs, tr, out_path_ssi)
        
        evi_np = cal_evi_index(src)
        out_path_ssi = os.path.join(out_folder, "evi_" + file_name)
        write_image(np.array([evi_np]), h, w, 1, crs, tr, out_path_ssi)        
        
            
"""Chay don anh"""    
    # img_path = r"C:\Users\SkyMap\Desktop\a.tif"
    # src, h, w, tr, crs = read_image(img_path)
    
    # avi_np = avi_index(src)
    # si_np = si_index(src)
    
    # out_avi = r"C:\Users\SkyMap\Desktop\avi.tif"
    # out_si = r"C:\Users\SkyMap\Desktop\si.tif"
    # write_image(np.array([avi_np]), h, w, 1, crs, tr, out_avi)
    # write_image(np.array([si_np]), h, w, 1, crs, tr, out_si)

