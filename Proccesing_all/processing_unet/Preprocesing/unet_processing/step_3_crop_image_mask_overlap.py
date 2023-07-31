# -*- coding: utf-8 -*-
from osgeo import gdal, gdalconst, ogr, osr
import numpy as np
import glob, os
import sys
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
import time
import argparse

core = multiprocessing.cpu_count()
def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.tif"):
        list_id.append(file[:-4])
        # print(file[:-4])
    return list_id

def crop_image(image_id,foder_image,foder_image_mask,true_size,overlap_size, path_image_crop,path_mask_crop):
    print(1)
    filename = os.path.join(foder_image,image_id+'.tif')
    dataset_image = gdal.Open(filename)
    data = dataset_image.ReadAsArray()
    img = np.array(data).swapaxes(0,1).swapaxes(1,2)
    (h,w) = img.shape[0:2]
    mask_path = os.path.join(foder_image_mask,image_id+'.tif')
    dataset_mask = gdal.Open(mask_path)
    list_hight_1 = list(range(0,h,overlap_size))
    list_weight_1 = list(range(0,w,overlap_size))
    list_hight = []
    list_weight = []
    for i in list_hight_1:
        if i < h - overlap_size:
            list_hight.append(i)        
    list_hight.append(h-overlap_size)
    
    for i in list_weight_1:
        if i < w - overlap_size:
            list_weight.append(i)        
    list_weight.append(w-overlap_size)
    
    count = 0
    for i in range(len(list_hight)):
        hight_tiles_up = list_hight[i]
#            hight_tiles_down = list_hight[i+1]
        for j in range(len(list_weight)):
            weight_tiles_up = list_weight[j]
    #                weight_tiles_down = list_weight[j+1]
            count = count+1
            output_image = os.path.join(path_image_crop,r'%s_%s.tif'%(image_id,str('{0:03}'.format(count))))
            # print(output_image)
            output_mask = os.path.join(path_mask_crop,r'%s_%s.tif'%(image_id,str('{0:03}'.format(count))))
            # r'D:\building\unetimages\weogeo_j284887\TIFF\%s\%s_%s.TIF'%(id_image,id_image,'{0:03}'.format(count))
            # output_mask = r'D:\building\unetimages\weogeo_j284887\TIFF\%s_MASK\%s_%s_mask.TIF'%(id_image,id_image,str('{0:03}'.format(count)))
            gdal.Translate(output_image, dataset_image,srcWin = [weight_tiles_up,hight_tiles_up,true_size,true_size])
            gdal.Translate(output_mask, dataset_mask,srcWin = [weight_tiles_up,hight_tiles_up,true_size,true_size])
    return True

def main(img_dir, mask_dir, crop_size, overlap_size):
    foder_name = os.path.basename(img_dir)        
    parent = os.path.dirname(img_dir)

    if not os.path.exists(os.path.join(parent,foder_name+'_crop')):
        os.makedirs(os.path.join(parent,foder_name+'_crop'))
    path_image_crop = os.path.join(parent,foder_name+'_crop')

    if not os.path.exists(os.path.join(parent,foder_name+'_mask_crop')):
        os.makedirs(os.path.join(parent,foder_name+'_mask_crop'))
    path_mask_crop = os.path.join(parent,foder_name+'_mask_crop')

    list_id = create_list_id(img_dir)
    p_cnt = Pool(processes=core)    
    result = p_cnt.map(partial(crop_image,foder_image=img_dir,foder_image_mask=mask_dir,true_size=crop_size,overlap_size=overlap_size,path_image_crop=path_image_crop,path_mask_crop = path_mask_crop), list_id)
    p_cnt.close()
    p_cnt.join()
    return path_image_crop,path_mask_crop

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
        '--img_dir',
        help='Image directory',
        required=True
    )


    args_parser.add_argument(
        '--mask_dir',
        help='mask directory',
        required=True
    )

    args_parser.add_argument(
        '--crop_size',
        help='Crop size (200 pixel) ',
        required=True
    )

    args_parser.add_argument(
        '--overlap_size',
        help='Overlap size (200 pixel) ',
        required=True
    )

    param = args_parser.parse_args()
    img_dir = param.img_dir
    mask_dir = param.mask_dir
    crop_size = int(param.crop_size)
    overlap_size = int(param.overlap_size)

    x1 = time.time()
    main(img_dir,mask_dir,crop_size,overlap_size)
    print(time.time() - x1, "second")