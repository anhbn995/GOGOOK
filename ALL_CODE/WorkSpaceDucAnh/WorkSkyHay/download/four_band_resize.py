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
import skimage.transform
import skimage.morphology
foder_path = os.path.abspath(sys.argv[1])
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

def buil_3_band(image_id,path_create):
    print(1)
    image_path = os.path.join(foder_path,image_id+'.tif')
    output = os.path.join(path_create,image_id+'.tif')
    ds = gdal.Open(image_path)
    options_list = [
    '-outsize 50.0% 50.0%',
    '-of GTiff',
    '-r cubicspline',
    '-ot Byte',
    '-co COMPRESS=LZW',
    '-co BIGTIFF=YES',
    '-colorinterp_4 undefined'
    ] 
    options_string = " ".join(options_list)

    gdal.Translate(output,
                image_path,
                options=options_string)
    return True
def main():
    list_id = create_list_id(foder_path)
    if not os.path.exists(os.path.join(parent,foder_name+'_resize')):
        os.makedirs(os.path.join(parent,foder_name+'_resize'))
    path_create = os.path.join(parent,foder_name+'_resize')
    p_cnt = Pool(processes=core)
    result = p_cnt.map(partial(buil_3_band,path_create=path_create), list_id)
    p_cnt.close()
    p_cnt.join()
if __name__=="__main__":
    x1 = time.time()
    main()
    print(time.time() - x1, "second")
