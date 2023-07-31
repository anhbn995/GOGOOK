import os
from osgeo import gdal, osr, ogr
import warnings
warnings.filterwarnings("ignore")

from download_and_processing_cloud.greencover_api.green_cover.utils import get_img_coords, write_results, padded_for_org_img, \
                                convert_value_obj, predict_water

# """Chay rieng cho dectet_green"""
# from utils import get_img_coords, write_results, padded_for_org_img, convert_value_obj, predict_water
# """Chay rieng cho dectet_green"""


def main_water(image_path, weight_path, result_path, model, input_size_water, 
        crop_size=100, batch_size=2, thresh_hold=0.15):
    print("*Init water model")
    model.load_weights(weight_path)

    print("*Predict image")
    thresh_hold = 1 - thresh_hold
    num_band = input_size_water[-1]
    input_size = input_size_water[0]
    
    dataset = gdal.Open(image_path)
    values = dataset.ReadAsArray()[0:num_band]
    h,w = values.shape[1:3]    
    padding = int((input_size - crop_size)/2)
    
    img_coords = get_img_coords(w, h, padding, crop_size)
    values = padded_for_org_img(values, num_band, padding)
    big_mask = predict_water(model, values, img_coords, num_band, h, w, padding, crop_size, 
                        input_size, batch_size, thresh_hold)

    mask_base = convert_value_obj(big_mask)

    write_results(mask_base, image_path, result_path)
    return mask_base

if __name__=="__main__":
    image_path = r"/home/skm/SKM16/Tmp/green_cover_change/2019-AOI1_fn.tif"
    weight_path = r"/home/skm/public_mount/DucAnhtmp/weights/water_weights.h5"
    result_path = r"/home/skm/SKM16/Tmp/green_cover_change/2019-AOI1_fn_rs_water.tif"
    n_labels = 1
    from models.models import unet_3plus
    input_size_water = [256, 256, 4]
    filter_num = [32, 64, 128, 256, 512]
    water_model = unet_3plus(input_size_water, n_labels, filter_num, filter_num_skip='auto', filter_num_aggregate='auto', 
                stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',batch_norm=True, 
                pool=True, unpool=True, deep_supervision=True,  multi_input=True ,backbone='ResNet50', 
                weights=None, freeze_backbone=False, freeze_batch_norm=True, name='unet3plus')

    main_water(image_path, weight_path, result_path, water_model, input_size_water, crop_size=100, batch_size=2, thresh_hold=0.15)