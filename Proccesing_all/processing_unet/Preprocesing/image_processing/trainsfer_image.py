# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 14:19:42 2018

@author: hue1
"""

from color_transfer import color_transfer
import gdal
import numpy as np
import time
import os
import sys
import cv2
#import h5py
#import scipy
image_source = os.path.abspath(sys.argv[1])
image_taget = os.path.abspath(sys.argv[2])
image_name = os.path.basename(image_taget)[:-4]
#size_crop = int(sys.argv[3])
parent = os.path.dirname(image_taget)

def main():
    ds1 = gdal.Open(image_source)
    data1 = ds1.ReadAsArray()
    img1 = np.array(data1).swapaxes(0,1).swapaxes(1,2)
    bgr1 =cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
#    with h5py.File(image_source, 'r') as f4:
#        X_mean = np.asarray(f4['x_mean'].value)
#    bandstats = {k: dict(max=0, min=0) for k in range(3)}
#    for i in range(3):
#        bandstats[i]['min'] = scipy.percentile(X_mean[i], 0)
#        bandstats[i]['max'] = scipy.percentile(X_mean[i], 100)
#    for chan_i in range(3):
#        min_val = bandstats[chan_i]['min']
#        max_val = bandstats[chan_i]['max']
#        X_mean[chan_i] = np.clip(X_mean[chan_i], min_val, max_val)
#        X_mean[chan_i] = (X_mean[chan_i] - min_val)/(max_val-min_val)*255
#    img_mean = np.array(X_mean).swapaxes(0,1).swapaxes(1,2)
#    img_mean = img_mean.astype(np.uint8)
    ds2 = gdal.Open(image_taget)
    data2 = ds2.ReadAsArray()
    img2 = np.array(data2).swapaxes(0,1).swapaxes(1,2)
    bgr2 =cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    transfer_bgr = color_transfer(bgr1, bgr2)
    transfer = cv2.cvtColor(transfer_bgr, cv2.COLOR_BGR2RGB)
    
    output = os.path.join(parent,image_name+'_transfer2.tif')
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(output,ds2.RasterXSize,ds2.RasterYSize,(transfer.shape[2]),gdal.GDT_Byte)#gdal.GDT_Byte/GDT_UInt16
    for i in range(1,transfer.shape[2]+1):
        dst_ds.GetRasterBand(i).WriteArray(transfer[:,:,i-1])
        dst_ds.GetRasterBand(i).ComputeStatistics(False)
    dst_ds.SetProjection(ds2.GetProjection())
    dst_ds.SetGeoTransform(ds2.GetGeoTransform())

if __name__=="__main__":
    x1 = time.time()
    main()
    print(time.time() - x1, "second")