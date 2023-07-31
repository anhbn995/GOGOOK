# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:14:08 2020

@author:ducanh
"""
import glob, os
from osgeo import gdal, gdalconst, ogr, osr
import numpy as np
import time
import rasterio

def cal_bancut(image_path, image_taken_min_max):
    dataset = rasterio.open(image_path)
    num_channel = dataset.count
    band_cut_th = {k: dict(max=0, min=0) for k in range(num_channel)}

    if image_taken_min_max:
        dataset_taken = rasterio.open(image_taken_min_max)
        for i_chan in range(num_channel):
            values_ = dataset_taken.read(i_chan+1).astype(np.float16)
            values_[values_==0] = np.nan
            band_cut_th[i_chan]['max'] = np.nanpercentile(values_, 98)
            band_cut_th[i_chan]['min'] = np.nanpercentile(values_, 2) 
    else:
        for i_chan in range(num_channel):
            values_ = dataset.read(i_chan+1).astype(np.float16)
            values_[values_==0] = np.nan
            band_cut_th[i_chan]['max'] = np.nanpercentile(values_, 98)
            band_cut_th[i_chan]['min'] = np.nanpercentile(values_, 2)
    return band_cut_th, num_channel

def buil_band(image_path, output, image_taken_min_max=None):
    band_cut_th, num_channel = cal_bancut(image_path, image_taken_min_max)
    options_list = ['-ot Byte','-a_nodata 0','-colorinterp_4 undefined']
    for i_chain in range(num_channel):
        options_list.append('-b {}'.format(i_chain+1))
    for i_chain in range(num_channel):
        options_list.append('-scale_{} {} {} 1.0 255.0 -exponent_{} 1.0'.format(i_chain+1,band_cut_th[i_chain]['min'],band_cut_th[i_chain]['max'],i_chain+1))

    options_string = " ".join(options_list)
    gdal.Translate(output,
            image_path,
            options=options_string)
    return True

def main():
    image = r"/media/skymap/SKM/Zang_zang_works/stack_image_chose_band/test_2.tif"
    out_path = r"/media/skymap/SKM/Zang_zang_works/stack_image_chose_band/test_byte_oke.tif"
    image_taken_min_max = None

    buil_band(image,out_path,image_taken_min_max)

if __name__=="__main__":
    x1 = time.time()
    main()
    print(time.time() - x1, "second")

