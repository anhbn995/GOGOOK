from rio_tiler.io import COGReader
 # -*- coding: utf-8 -*-
import glob, os
from osgeo import gdal
import sys
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
import time
import rasterio
import numpy as np


foder_path = os.path.abspath(sys.argv[1])
num_channel = int((sys.argv[2]))


parent = os.path.dirname(foder_path)
foder_name = os.path.basename(foder_path)
core = multiprocessing.cpu_count()//6

def create_list_id(path):
    list_image = []
    for root, dirs, files in os.walk(path):
        print(dirs)
        for file in files:
            if file.endswith(".tif"):
                list_image.append(os.path.join(root, file))
    return list_image
    
def get_mean_by_quantile_schema(input_path):
    qt_scheme = {}
    try:
        with COGReader(input_path) as cog:
            stats = cog.stats()
            idx = 0
            for _, value in stats.items():
                qt_scheme[idx] = {
                    'max': value['percentiles'][1],
                    'min': value['percentiles'][0],
                    'mean': value['mean'],
                    'std': value['std']
                }
            idx += 1
    except:
        with COGReader(input_path) as cog:
            stats = cog.statistics()
            idx = 0
            for _, value in stats.items():
                qt_scheme[idx] = {
                    'max': value['percentile_98'],
                    'min': value['percentile_2'],
                    'mean': value['mean'],
                    'std': value['std']
                }
                idx +=1
    mean_each_band = list()
    band_stt = list()
    for key in qt_scheme:
        band_stt.append(key)
        mean_each_band.append(qt_scheme[key]['mean'])
    return np.array(mean_each_band), tuple(np.array(band_stt) + 1)



def cal_mean_std_by_quantile_schema(dir_img, band_stt):
    list_fp_img = glob.glob(os.path.join(dir_img, '*.tif'))
    # tinh mean all
    mean_all = 0.
    for fp_img in list_fp_img:
        mean, _ = get_mean_by_quantile_schema(fp_img)
        mean_all += mean
    mean_all /= len(list_fp_img)
    print(mean_all)
    
    total_pixel = 0
    for fp_img in list_fp_img:
        with rasterio.open(fp_img) as src:
            img = src.read(band_stt)
            total_pixel += src.height*src.width
        
        deviations = img - 
        
        
        
    
    
    
    




def buil_3_band(image_path,path_create,num_channel):
    # print(1)
    dir_name = os.path.basename(os.path.dirname(image_path))
    image_name = os.path.basename(image_path)[:-4]
    path_out = os.path.join(path_create,dir_name)
    band_cut_th = get_quantile_schema(image_path)
    options_list = ['-ot Byte','-a_nodata 0','-colorinterp_4 undefined']
    for i_chain in range(num_channel):
        options_list.append('-b {}'.format(i_chain+1))
    for i_chain in range(num_channel):
        options_list.append('-scale_{} {} {} 1.0 255.0 -exponent_{} 1.0'.format(i_chain+1,band_cut_th[i_chain]['min'],band_cut_th[i_chain]['max'],i_chain+1))

    output = os.path.join(path_out,image_name+'.tif')
    options_string = " ".join(options_list)
    print(band_cut_th)
    gdal.Translate(output,
            image_path,
            options=options_string)
    print(options_list)
    return True
def main():
    list_id = create_list_id(foder_path)
    print(os.getcwd())
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
