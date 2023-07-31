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
foder_image_mask = os.path.abspath(sys.argv[2])
foder_name = os.path.basename(foder_image)
size_crop = int(sys.argv[3])
true_size = int(sys.argv[4])
parent = os.path.dirname(foder_image)
core = multiprocessing.cpu_count()
def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.tif"):
        list_id.append(file[:-4])
        # print(file[:-4])
    return list_id

def crop_image(image_id,path_image_crop,path_mask_crop):
    print(1)
    filename = os.path.join(foder_image,image_id+'.tif')
    dataset_image = gdal.Open(filename)
    data = dataset_image.ReadAsArray()
    img = np.array(data).swapaxes(0,1).swapaxes(1,2)
    (h,w) = img.shape[0:2]
    mask_path = os.path.join(foder_image_mask,image_id+'.tif')
    dataset_mask = gdal.Open(mask_path)
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
            output_mask = os.path.join(path_mask_crop,r'%s_%s.tif'%(image_id,str('{0:03}'.format(count))))
            # r'D:\building\unetimages\weogeo_j284887\TIFF\%s\%s_%s.TIF'%(id_image,id_image,'{0:03}'.format(count))
            # output_mask = r'D:\building\unetimages\weogeo_j284887\TIFF\%s_MASK\%s_%s_mask.TIF'%(id_image,id_image,str('{0:03}'.format(count)))
            gdal.Translate(output_image, dataset_image,srcWin = [weight_tiles_up,hight_tiles_up,size_crop,size_crop])
            gdal.Translate(output_mask, dataset_mask,srcWin = [weight_tiles_up,hight_tiles_up,size_crop,size_crop])
    return True

def main():
    if not os.path.exists(os.path.join(parent,foder_name+'_crop')):
        os.makedirs(os.path.join(parent,foder_name+'_crop'))
    path_image_crop = os.path.join(parent,foder_name+'_crop')

    if not os.path.exists(os.path.join(parent,foder_name+'_mask_crop')):
        os.makedirs(os.path.join(parent,foder_name+'_mask_crop'))
    path_mask_crop = os.path.join(parent,foder_name+'_mask_crop')

    list_id = create_list_id(foder_image)
    p_cnt = Pool(processes=core)
    result = p_cnt.map(partial(crop_image,path_image_crop=path_image_crop,path_mask_crop = path_mask_crop), list_id)
    p_cnt.close()
    p_cnt.join()
    return True
if __name__ == "__main__":
    x1 = time.time()
    main()
    print(time.time() - x1, "second")