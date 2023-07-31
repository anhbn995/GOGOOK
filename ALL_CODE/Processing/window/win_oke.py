#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 09:26:05 2021

@author: skm
"""
import gdal
import rasterio
import numpy as np
from rasterio.windows import Window

image_path  = r"/home/skm/SKM/xoa_nhe_3band.tif"
outputFileName = r"/home/skm/SKM/manh6.tif"
input_size = 500
crop_size = 400

def write_window_many_chanel(output_ds, arr_c, s_h, e_h ,s_w, e_w, 
                                               sw_w, sw_h, size_w_crop, size_h_crop):
    for c, arr in enumerate(arr_c):
        output_ds.write(arr[s_h:e_h,s_w:e_w],window = Window(sw_w, sw_h, size_w_crop, size_h_crop), indexes= c + 1)


def predict(arr):
    return arr + 1


with rasterio.open(image_path) as src:
    h,w = src.height,src.width
    source_crs = src.crs
    source_transform = src.transform
    dtype_or = src.dtypes
    num_band = src.count
    
with rasterio.open(outputFileName, 'w', driver='GTiff',
                            height = h, width = w,
                            count=num_band, dtype='float64',
                            crs=source_crs,
                            transform=source_transform,
                            nodata=0,
                            compress='lzw') as output_ds:
    output_ds = np.empty((num_band,h,w))
    
    
padding = int((input_size - crop_size)/2)
list_weight = list(range(0, w, crop_size))
list_hight = list(range(0, h, crop_size))

print(list_weight, list_hight)
# ds_image = gdal.Open(image_path, GA_ReadOnly)
src = rasterio.open(image_path)

with rasterio.open(outputFileName,"r+") as output_ds:
    for start_h_org in list_hight:
        for start_w_org in list_weight:
            # vi tri bat dau
            h_crop_start = start_h_org - padding
            w_crop_start = start_w_org - padding
            # kich thuoc
            tmp_img_size_model = np.zeros((num_band, input_size,input_size))
            # truong hop 0 0
            if h_crop_start < 0 and w_crop_start < 0:
                # continue
                h_crop_start = 0
                w_crop_start = 0
                size_h_crop = crop_size + padding
                size_w_crop = crop_size + padding
                img_window_crop  = src.read(window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
                tmp_img_size_model[:, padding:, padding:] = img_window_crop
                img_predict = predict(tmp_img_size_model)
                write_window_many_chanel(output_ds, img_predict, padding, crop_size + padding, padding, crop_size + padding, 
                                                                start_w_org, start_h_org, crop_size, crop_size)
            
            # truong hop h = 0 va w != 0
            elif h_crop_start < 0:
                h_crop_start = 0
                size_h_crop = crop_size + padding
                size_w_crop = min(crop_size + 2*padding, w - start_w_org + padding)
                img_window_crop  = src.read(window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
                
                if size_w_crop == w - start_w_org + padding:
                    end_c_index_w =  size_w_crop
                    tmp_img_size_model[:,padding:,:end_c_index_w] = img_window_crop
                else:
                    end_c_index_w = crop_size + padding
                    tmp_img_size_model[:, padding:,:] = img_window_crop
                img_predict = predict(tmp_img_size_model)
                write_window_many_chanel(output_ds, img_predict, padding, crop_size + padding ,padding, end_c_index_w, 
                                            start_w_org, start_h_org,  min(crop_size, w - start_w_org), crop_size)
            
            # Truong hop w = 0, h!=0 
            elif w_crop_start < 0:
                w_crop_start = 0
                size_w_crop = crop_size + padding
                size_h_crop = min(crop_size + 2*padding, h - start_h_org + padding)
                img_window_crop  = src.read(window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
                
                if size_h_crop == h - start_h_org + padding:
                    end_c_index_h =  size_h_crop
                    tmp_img_size_model[:,:end_c_index_h,padding:] = img_window_crop
                else:
                    end_c_index_h = crop_size + padding
                    tmp_img_size_model[:,:, padding:] = img_window_crop
                img_predict = predict(tmp_img_size_model)
                write_window_many_chanel(output_ds, img_predict, padding, end_c_index_h, padding, crop_size + padding, 
                                            start_w_org, start_h_org, crop_size, min(crop_size, h - start_h_org))
                
            # Truong hop ca 2 deu khac khong
            else:
                        size_w_crop = min(crop_size +2*padding, w - start_w_org + padding)
                        size_h_crop = min(crop_size +2*padding, h - start_h_org + padding)
                        img_window_crop  = src.read(window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
                        # print(img_window_crop.shape, size_w_crop, size_h_crop)
                        if size_w_crop < (crop_size + 2*padding) and size_h_crop < (crop_size + 2*padding):
                            print(img_window_crop.shape, size_w_crop, size_h_crop)
                            end_c_index_h = size_h_crop
                            end_c_index_w = size_w_crop
                            tmp_img_size_model[:,:end_c_index_h,:   end_c_index_w] = img_window_crop
                        elif size_w_crop < (crop_size + 2*padding):
                            end_c_index_h = crop_size + padding
                            end_c_index_w = size_w_crop
                            tmp_img_size_model[:,:,:end_c_index_w] = img_window_crop
                        elif size_h_crop < (crop_size + 2*padding):
                            end_c_index_w = crop_size + padding
                            end_c_index_h = size_h_crop
                            tmp_img_size_model[:,:end_c_index_h,:] = img_window_crop
                        else:
                            end_c_index_w = crop_size + padding
                            end_c_index_h = crop_size + padding
                            tmp_img_size_model[:,:,:] = img_window_crop
                        img_predict = predict(tmp_img_size_model) 
                        write_window_many_chanel(output_ds, img_predict, padding, end_c_index_h, padding, end_c_index_w, 
                                                    start_w_org, start_h_org, min(crop_size, w - start_w_org), min(crop_size, h - start_h_org))
    
                        
                            


                    
                    
    



                
             




            

