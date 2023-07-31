# -*- coding: utf-8 -*-
import glob, os
from osgeo import gdal, gdalconst, ogr, osr
import numpy as np
import math
import sys
from pyproj import Proj, transform
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
import time
import scipy
foder_path = os.path.abspath(sys.argv[1])
num_channel = int((sys.argv[2]))
parent = os.path.dirname(foder_path)
foder_name = os.path.basename(foder_path)
core = multiprocessing.cpu_count()
def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.tif"):
        list_id.append(file[:-4])
        # print(file[:-4])
    return list_id

def cal_bancut(image_id,num_channel):
    band_values = {k: [] for k in range(4)}
    image_path =  os.path.join(foder_path,image_id + ".tif")
    print(image_path)
    dataset = gdal.Open(image_path)
    data = dataset.ReadAsArray()
    for i_chan in range(num_channel):
        values_ = data[i_chan].ravel().tolist()
        # values_ = np.array(
        #     [v for v in values_ if v != 0]
        # )  # Remove sensored mask
        band_values[i_chan].append(values_)
    band_cut_th = {k: dict(max=0, min=0) for k in range(4)}
    for i_chan in range(num_channel):
        band_values[i_chan] = np.concatenate(
            band_values[i_chan]).ravel()
        band_cut_th[i_chan]['max'] = np.percentile(
            band_values[i_chan], 98)
        band_cut_th[i_chan]['min'] = np.percentile(
            band_values[i_chan], 2)
    print(band_cut_th)
    # np.save('band_cut',band_cut_th)
    # with open('bandcut.txt', 'w') as outfile:  
    #     json.dump(band_cut_th, outfile)
    return band_cut_th
def buil_3_band(image_id,path_create,num_channel):
    print(1)
    image_path = os.path.join(foder_path,image_id+'.tif')
    band_cut_th = cal_bancut(image_id,num_channel)
    options_list = ['-ot Byte']
    for i_chain in range(num_channel):
        options_list.append('-b {}'.format(i_chain+1))
    for i_chain in range(num_channel):
        options_list.append('-scale_{} {} {} 0 255'.format(i_chain+1,band_cut_th[i_chain]['min'],band_cut_th[i_chain]['max']))
    # options_list = [
    #     '-ot Byte',
    #     '-b 1',
    #     '-b 2',
    #     '-b 3',
    #     '-b 4',
    #     '-scale_1 {} {} 0 255',
    #     '-scale_2 {} {} 0 255',
    #     '-scale_3 {} {} 0 255',
    #     '-scale_4 427.0 3195.0 0 255'
    #     ] 
    output = os.path.join(path_create,image_id+'.tif')
    options_string = " ".join(options_list)
    gdal.Translate(output,
            image_path,
            options=options_string)
    # gdal_translate -ot Byte -b 3 -b 2 -b 1 -scale_1 880 1400 0 255 -scale_2 660 1300 0 255 -scale_3 480 1300 0 255 D:\data_source\GoogleEvent\2018\2018\%%~nf.tif %%~nf_rgb.tif
    # img2 = np.array(data3).swapaxes(0,1).swapaxes(1,2)*255
    # output = os.path.join(path_create,image_id+'.tif')
    # driver = gdal.GetDriverByName("GTiff")
    # dst_ds = driver.Create(output,ds.RasterXSize,ds.RasterYSize,(img2.shape[2]),gdal.GDT_Byte)#gdal.GDT_Byte/GDT_UInt16
    # for i in range(1,img2.shape[2]+1):
    #     dst_ds.GetRasterBand(i).WriteArray(img2[:,:,i-1])
    #     dst_ds.GetRasterBand(i).ComputeStatistics(False)
    # dst_ds.SetProjection(ds.GetProjection())
    # dst_ds.SetGeoTransform(ds.GetGeoTransform())
    return True
def main():
    list_id = create_list_id(foder_path)
    if not os.path.exists(os.path.join(parent,foder_name+'_8bit_perimage')):
        os.makedirs(os.path.join(parent,foder_name+'_8bit_perimage'))
    path_create = os.path.join(parent,foder_name+'_8bit_perimage')
    p_cnt = Pool(processes=core)
    result = p_cnt.map(partial(buil_3_band,path_create=path_create,num_channel=num_channel), list_id)
    p_cnt.close()
    p_cnt.join()
if __name__=="__main__":
    x1 = time.time()
    main()
    print(time.time() - x1, "second")