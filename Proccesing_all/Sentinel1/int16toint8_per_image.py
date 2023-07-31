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
    list_image = []
    # files = os.listdir(path)
    # for dir_name in files:
    #     for 
    # return files
    for root, dirs, files in os.walk(path):
        print(dir)
        for file in files:
            if file.endswith(".tif"):
                list_image.append(os.path.join(root, file))
    return list_image
    
def cal_bancut(image_path,num_channel):
    
    dataset = gdal.Open(image_path)
    band_cut_th = {k: dict(max=0, min=0) for k in range(num_channel)}
    for i_chan in range(num_channel):
        values_ = dataset.GetRasterBand(i_chan+1).ReadAsArray().astype(np.float16)
        values_[values_==0] = np.nan
        band_cut_th[i_chan]['max'] = np.nanpercentile(
            values_, 98)
        band_cut_th[i_chan]['min'] = np.nanpercentile(
            values_, 2)
    print(band_cut_th)
    # np.save('band_cut',band_cut_th)
    # with open('bandcut.txt', 'w') as outfile:  
    #     json.dump(band_cut_th, outfile)
    return band_cut_th
def buil_3_band(image_path,path_create,num_channel):
    print(1)
    dir_name = os.path.basename(os.path.dirname(image_path))
    image_name = os.path.basename(image_path)[:-4]
    path_out = os.path.join(path_create,dir_name)
    band_cut_th = cal_bancut(image_path,num_channel)
    options_list = ['-ot Byte','-a_nodata 0','-colorinterp_4 undefined']
    for i_chain in range(num_channel):
        options_list.append('-b {}'.format(i_chain+1))
    for i_chain in range(num_channel):
        options_list.append('-scale_{} {} {} 1.0 255.0 -exponent_{} 1.0'.format(i_chain+1,band_cut_th[i_chain]['min'],band_cut_th[i_chain]['max'],i_chain+1))
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
    output = os.path.join(path_out,image_name+'.tif')
    options_string = " ".join(options_list)
    print(options_string)
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
    print(os.getcwd())
    # #list_id = ["suratsample1_Pre data"]
    if not os.path.exists(os.path.join(parent,foder_name+'_8bit_perimage')):
        os.makedirs(os.path.join(parent,foder_name+'_8bit_perimage'))
    path_create = os.path.join(parent,foder_name+'_8bit_perimage')

    for image_path1 in list_id:
        dir_name = os.path.basename(os.path.dirname(image_path1))
        if not os.path.exists(os.path.join(path_create,dir_name)):
            os.makedirs(os.path.join(path_create,dir_name))
    p_cnt = Pool(processes=core)
    result = p_cnt.map(partial(buil_3_band,path_create=path_create,num_channel=num_channel), list_id)
    p_cnt.close()
    p_cnt.join()
if __name__=="__main__":
    x1 = time.time()
    main()
    print(time.time() - x1, "second")
