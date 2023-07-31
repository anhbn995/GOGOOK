# -*- coding: utf-8 -*-
from osgeo import gdal, gdalconst, ogr, osr
import numpy as np
import shapefile as shp
from pyproj import Proj, transform
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib.colors import rgb_to_hsv
from math import pi
import glob, os
import sys
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
import time
foder_image = os.path.abspath(sys.argv[1])
foder_dsm = os.path.abspath(sys.argv[2])
foder_dtm = os.path.abspath(sys.argv[3])
foder_name = os.path.basename(foder_image)
size_crop = int(sys.argv[4])
true_size = int(sys.argv[5])
parent = os.path.dirname(foder_image)
core = multiprocessing.cpu_count()
def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.tif"):
        list_id.append(file[:-4])
        # print(file[:-4])
    return list_id

def crop_image(image_id,path_image_crop,path_dsm_crop,path_dtm_crop):
    print(1)
    filename = os.path.join(foder_image,image_id+'.tif')
    dataset_image = gdal.Open(filename)
    w,h = dataset_image.RasterXSize, dataset_image.RasterYSize
    # data = dataset_image.ReadAsArray()
    # img = np.array(data).swapaxes(0,1).swapaxes(1,2)
    # (h,w) = img.shape[0:2]
    dsm_path = os.path.join(foder_dsm,image_id+'.tif')
    dataset_dsm = gdal.Open(dsm_path)
    dtm_path = os.path.join(foder_dtm,image_id+'.tif')
    dataset_dtm = gdal.Open(dtm_path)
    list_hight_1 = list(range(0,h,true_size))
    list_weight_1 = list(range(0,w,true_size))
    list_hight = []
    list_weight = []
    for i in list_hight_1:
        if i < h - size_crop:
            list_hight.append(i)        
    list_hight.append(h-size_crop)
    
    for i in list_weight_1:
        if i < w - size_crop:
            list_weight.append(i)        
    list_weight.append(w-size_crop)
    
    count = 0
    for i in range(len(list_hight)):
        hight_tiles_up = list_hight[i]
#            hight_tiles_down = list_hight[i+1]
        for j in range(len(list_weight)):
            weight_tiles_up = list_weight[j]
    #                weight_tiles_down = list_weight[j+1]
            count = count+1
            output_image = os.path.join(path_image_crop,r'%s_%s.tif'%(image_id,str('{0:03}'.format(count))))
            print(output_image)
            output_dsm = os.path.join(path_dsm_crop,r'%s_%s.tif'%(image_id,str('{0:03}'.format(count))))
            output_dtm = os.path.join(path_dtm_crop,r'%s_%s.tif'%(image_id,str('{0:03}'.format(count))))
            # r'D:\building\unetimages\weogeo_j284887\TIFF\%s\%s_%s.TIF'%(id_image,id_image,'{0:03}'.format(count))
            # output_mask = r'D:\building\unetimages\weogeo_j284887\TIFF\%s_MASK\%s_%s_mask.TIF'%(id_image,id_image,str('{0:03}'.format(count)))
            gdal.Translate(output_image, dataset_image,srcWin = [weight_tiles_up,hight_tiles_up,size_crop,size_crop])
            gdal.Translate(output_dsm, dataset_dsm,srcWin = [weight_tiles_up,hight_tiles_up,size_crop,size_crop])
            gdal.Translate(output_dtm, dataset_dtm,srcWin = [weight_tiles_up,hight_tiles_up,size_crop,size_crop])
    return True

def main():
    path_image_crop = os.path.join(parent,'data_crop',"images")
    if not os.path.exists(path_image_crop):
        os.makedirs(path_image_crop)
    
    path_dsm_crop = os.path.join(parent,'data_crop',"dsm")
    if not os.path.exists(path_dsm_crop):
        os.makedirs(path_dsm_crop)
    path_dtm_crop = os.path.join(parent,'data_crop',"dtm")
    if not os.path.exists(path_dtm_crop):
        os.makedirs(path_dtm_crop)

    list_id = create_list_id(foder_image)
    p_cnt = Pool(processes=core)
    result = p_cnt.map(partial(crop_image,path_image_crop=path_image_crop,path_dsm_crop = path_dsm_crop,path_dtm_crop=path_dtm_crop), list_id)
    p_cnt.close()
    p_cnt.join()
    return True
if __name__ == "__main__":
    x1 = time.time()
    main()
    print(time.time() - x1, "second")