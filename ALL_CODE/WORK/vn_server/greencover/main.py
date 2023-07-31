import os
import glob
import rasterio
import uuid
import numpy as np
from forest import main_only_forest
from green_cover import run_segmentation


def sum_class(list_fp, index_nodata, out_path):
    with rasterio.open(list_fp[0]) as src:
        img_green_cover = src.read()[0]
        meta = src.meta
    
    img_forest = np.zeros_like(img_green_cover)
    if len(list_fp) < 3:
        with rasterio.open(list_fp[1]) as src:
            img_water = src.read()[0]
            
    else:
        with rasterio.open(list_fp[1]) as src:
            img_water = src.read()[0]
        with rasterio.open(list_fp[2]) as src:
            img_forest = src.read()[0]  
        
    index_water = np.where(img_water != 0)
    index_forest = np.where(img_forest != 0)
    del img_water, img_forest
    
    img_green_cover[index_water] = 2
    img_green_cover[index_forest] = 3
    
    color = {
                0: (0,0,0, 0),
                1: (0,255,0,0),      #Green
                2: (100, 149, 237, 0), #water
                
            }
    if 3 in img_green_cover:
        img_green_cover[img_green_cover==0] = 4
        color.update(
            {
                3: (0,128,0),     #Forest
                4: (101,67,33, 0) #Buildup
            })
    else:
        img_green_cover[img_green_cover==0] = 3
        color.update(
            {
                3: (101,67,33, 0),  #Buildup
            })
          
    img_green_cover[index_nodata] = 0
    
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(np.array([img_green_cover]))
        dst.write_colormap(1, color) 
        
  
def main(fp_in, fp_out, dir_result_path_green_and_water, weight_path_green=None, weight_path_water=None, weight_path_forest=None):
    os.makedirs(dir_result_path_green_and_water, exist_ok=True)
    rs_green, rs_water, index_nodata = run_segmentation(fp_in, weight_path_green, weight_path_water, dir_result_path_green_and_water, dil=False, run_agian=False)
    
    list_fp_rs = [rs_green, rs_water]
    if weight_path_forest:
        rs_forest = main_only_forest(use_model = 'att_unet', weight_path = weight_path_forest, 
                        image_path = fp_in, result_dir = dir_result_path_green_and_water)
        list_fp_rs = [rs_green, rs_water, rs_forest]
    sum_class(list_fp_rs, index_nodata, fp_out)


if __name__ == '__main__':
    weight_path_green = "/home/skm/SKM/WORK/ALL_CODE/WORK/greencover_api/weights/green_weights.h5"
    weight_path_water = "/home/skm/SKM/WORK/ALL_CODE/WORK/greencover_api/weights/water_weights.h5"
    weight_path_forest ="/home/skm/SKM/WORK/ALL_CODE/WORK/greencover_api/weights/forest_weights_v2.h5"
    
    # run 1 image
    # fp_in = "/home/skm/SKM16/Work/GreenCover_World/melbourne/2021.tif"
    # dir_result_path_green_and_water = f'/home/skm/SKM16/Work/GreenCover_World/tmp'
    # fp_out = "/home/skm/SKM16/Work/GreenCover_World/aaa.tif"
    # main(fp_in, fp_out, dir_result_path_green_and_water, weight_path_green, weight_path_water, weight_path_forest)
    
    # run folder image
    dir_in = '/home/skm/SKM16/Work/GreenCover_World/san-jose/mosaic'
    dir_rs_tmp = '/home/skm/SKM16/Work/GreenCover_World/san-jose/mosaic/tmp'
    dir_out = '/home/skm/SKM16/Work/GreenCover_World/san-jose/mosaic/rs'
    os.makedirs(dir_out, exist_ok=True)
    
    list_fp_in = glob.glob(os.path.join(dir_in, '*.tif'))
    for fp_in in list_fp_in:
        dir_rs_w_g_f_tmp = os.path.join(dir_rs_tmp, uuid.uuid4().hex)
        os.makedirs(dir_rs_w_g_f_tmp, exist_ok=True)
        fp_out = os.path.join(dir_out, os.path.basename(fp_in).replace('.tif','greencover.tif'))
        main(fp_in, fp_out, dir_rs_w_g_f_tmp, weight_path_green, weight_path_water, weight_path_forest)