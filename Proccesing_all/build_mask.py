import rasterio
import rasterio.features
import os
import geopandas as gp
import numpy as np
from matplotlib import pyplot as plt
import cv2
import argparse
import sys
import glob, os
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
import time
core = multiprocessing.cpu_count()


def create_mask_by_shapefile2(fp_shp, height, width, tr, crs):
    list_geometry = []
    crs_img = crs.to_string()
    shp = gp.read_file(fp_shp)
    shp = shp.to_crs(crs_img)
    ls_geo = [(x.geometry) for i, x in shp.iterrows()]
    list_geometry.extend(ls_geo)
    mask = rasterio.features.rasterize(list_geometry
                                    ,out_shape=(height, width)
                                    ,transform=tr)
    return mask

def arr2raster(path_out, bands, height, width, tr, dtype="uint8",coordinate=None,projstr=None):
    num_band = len(bands)
    if coordinate!= None:
        crs = rasterio.crs.CRS.from_epsg(coordinate)
    else:
        crs = rasterio.crs.CRS.from_string(projstr)
    new_dataset = rasterio.open(path_out, 'w', driver='GTiff',
                            height = height, width = width,
                            count = num_band, dtype = dtype,
                            crs = crs,
                            transform = tr,
                            # nodata = 0,
                            compress='lzw')
    if num_band == 1:
        new_dataset.write(bands[0], 1)
    else:
        for i in range(num_band):
            new_dataset.write(bands[i],i+1)
    new_dataset.close()


def build_mask(image_id,img_dir,path_create,folder_shape):
    path_image = os.path.join(img_dir,image_id+'.tif')
    output_mask =  os.path.join(path_create,image_id+'.tif')
    path_shape = os.path.join(folder_shape, image_id+'.shp')

    with rasterio.open(path_image) as src:
        tr = src.transform
        w,h = src.width,src.height
        projstr = (src.crs.to_string())
        print(projstr)
        crs = src.crs
        check_epsg = crs.is_epsg_code
        coordinate = src.crs.to_epsg()
    mask1 = create_mask_by_shapefile2(path_shape, h, w, tr, crs)*255
    arr2raster(output_mask, [mask1], h, w, tr, dtype="uint8",coordinate=coordinate,projstr=projstr)

def create_list_id(path,end_str):
    num_str = len(end_str)
    list_id = []
    os.chdir(path)
    for file in glob.glob("*{}".format(end_str)):
        list_id.append(file[:-num_str])
        # print(file[:-4])
    return list_id

def main_build_mask(img_dir, folder_shape):    
    parent = os.path.dirname(img_dir)
    foder_name = os.path.basename(img_dir)

    list_id = create_list_id(img_dir,'.tif')
    if not os.path.exists(os.path.join(parent,foder_name+'_mask')):
        os.makedirs(os.path.join(parent,foder_name+'_mask'))
    path_create = os.path.join(parent,foder_name+'_mask')

    p_cnt = Pool(processes=core)
    result = p_cnt.map(partial(build_mask,img_dir=img_dir,path_create=path_create,folder_shape=folder_shape), list_id)
    p_cnt.close()
    p_cnt.join()
    return path_create

if __name__ == "__main__":
    # args_parser = argparse.ArgumentParser()

    # args_parser.add_argument(
    #     '--img_dir',
    #     help='Orginal Image Directory',
    #     required=True
    # )


    # args_parser.add_argument(
    #     '--shape_dir',
    #     help='Box cut directory',
    #     required=True
    # )

    # param = args_parser.parse_args()
    # img_dir = param.img_dir
    # shape_path = param.shape_dir


    img_dir = os.path.abspath(sys.argv[1])
    shape_path = os.path.abspath(sys.argv[2]) 

    # img_dir = r"/mnt/66A8E45DA8E42CED/farm_singlefarm/img_cut_box"
    # shape_path = r"/mnt/66A8E45DA8E42CED/farm_singlefarm/shp_farm"

    x1 = time.time()
    main_build_mask(img_dir,shape_path)
    print(time.time() - x1, "second")