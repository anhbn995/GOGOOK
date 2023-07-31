import os
from posixpath import basename
from turtle import pd
import rasterio
import numpy as np

from download_and_processing_cloud.greencover_api.green_cover.detect_green import main_green
from download_and_processing_cloud.greencover_api.green_cover.detect_water import main_water
from download_and_processing_cloud.greencover_api.green_cover.models.models import unet_3plus, att_unet

def combine_all(path_image, result_green, result_water, result_path, tmp_folder):
    result_color = os.path.join(result_path, os.path.basename(path_image))
    tmp_results = os.path.join(tmp_folder, os.path.basename(path_image))

    with rasterio.open(path_image) as src:
        msk = src.read_masks(1)
        out_meta = src.meta
    msk = msk + 1
    # msk = msk*5
    if type(result_green) == bool:
        mask_green = rasterio.open(tmp_results.replace('.tif','_green.tif')).read()[0]
    else:
        mask_green = result_green

    if type(result_water) == bool:
        mask_water = rasterio.open(tmp_results.replace('.tif','_water.tif')).read()[0]
        mask_water[mask_water==255]=1
    else:
        mask_water = result_water

    mask_water[mask_water==1]=2
    mask_water[mask_green==1]=1
    mask_water[mask_water==0]=3
    mask_water[msk==1]=0
    mask_all = mask_water
    combine_path = tmp_results.replace('.tif','_combine.tif')
    with rasterio.Env():
        profile = out_meta
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw')
        with rasterio.open(combine_path, 'w', **profile) as dst:
            dst.write(mask_all.astype(np.uint8),1)

    with rasterio.Env():   
        with rasterio.open(combine_path) as src:  
            shade = src.read()[0]
            meta = src.meta.copy()
            meta.update({'nodata': 0, 'dtype':'uint8'})
        with rasterio.open(result_color, 'w', **meta) as dst:
            dst.write(shade, indexes=1)
            dst.write_colormap(1, {
                    0: (0,0,0, 0),
                    1: (34,139,34,0), #Green
                    2: (100, 149, 237, 0), #water
                    3: (101,67,33, 0)}) #Buildup
        return result_color

def run_segmentation(img, weight_path_green, weight_path_water, result_path, dil=False, run_agian=False):
    n_labels = 1
    input_size_green = [128, 128, 4]
    filter_num = [128, 256, 512, 1024]
    green_model = att_unet(input_size_green, filter_num, n_labels, stack_num_down=2, stack_num_up=2, activation='ReLU', 
            atten_activation='ReLU', attention='add', output_activation='Sigmoid', batch_norm=False, 
            pool=True, unpool=True, backbone=None, weights='imagenet', freeze_backbone=True,
            freeze_batch_norm=True, name='attunet')

    input_size_water = [256, 256, 4]
    filter_num = [32, 64, 128, 256, 512]
    water_model = unet_3plus(input_size_water, n_labels, filter_num, filter_num_skip='auto', filter_num_aggregate='auto', 
                stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',batch_norm=True, 
                pool=True, unpool=True, deep_supervision=True,  multi_input=True ,backbone='ResNet50', 
                weights=None, freeze_backbone=False, freeze_batch_norm=True, name='unet3plus')

    result_path = os.path.join(result_path, os.path.basename(img))
    result_green = result_path.replace('.tif', '_green.tif')
    if not os.path.exists(result_green):
        tmp_green = main_green(img, weight_path_green, result_green, green_model, input_size_green, dil=dil)
    else:
        if run_agian:
            tmp_green = main_green(img, weight_path_green, result_green, green_model, input_size_green, dil=dil)
        else:
            tmp_green = False
            pass
    result_water = result_path.replace('.tif', '_water.tif')
    if not os.path.exists(result_water):
        tmp_water = main_water(img, weight_path_water, result_water, water_model, input_size_water)
    else:
        if run_agian:
            tmp_water = main_water(img, weight_path_water, result_water, water_model, input_size_water)
        else:
            tmp_water = False
    return tmp_green, tmp_water

