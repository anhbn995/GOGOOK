import os
from unicodedata import name
import cv2
from osgeo import gdal, osr, ogr

import warnings
warnings.filterwarnings("ignore")

from download_and_processing_cloud.greencover_api.green_cover.utils import get_img_coords, write_results, padded_for_org_img, \
                                convert_value_obj, predict_green


# """Chay rieng cho dectet_green"""
# from utils import get_img_coords, write_results, padded_for_org_img, \
#                                 convert_value_obj, predict_green
# """Chay rieng cho dectet_green"""

def main_green(image_path, weight_path, result_path, model, input_size_green, dil, 
        crop_size=100, batch_size=2, thresh_hold=0.15):
    print("*Init green model")
    model.load_weights(weight_path)

    print("*Predict image")
    thresh_hold = 1 - thresh_hold
    num_band = input_size_green[-1]
    input_size = input_size_green[0]
    
    dataset = gdal.Open(image_path)
    values = dataset.ReadAsArray()[0:num_band]
    h,w = values.shape[1:3]    
    padding = int((input_size - crop_size)/2)
    
    img_coords = get_img_coords(w, h, padding, crop_size)
    values = padded_for_org_img(values, num_band, padding)
    big_mask = predict_green(model, values, img_coords, num_band, h, w, padding, crop_size, 
                        input_size, batch_size, thresh_hold)

    mask_base = convert_value_obj(big_mask)
    
    if dil:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        mask_base = cv2.dilate(mask_base,kernel,iterations = 1)
    else:
        mask_base = mask_base

    write_results(mask_base, image_path, result_path)
    return mask_base


if __name__=="__main__":
    image_path = r"/home/skm/SKM16/Tmp/green_cover_change/2022-AOI1_fn.tif"
    weight_path = r"/home/skm/public_mount/DucAnhtmp/weights/green_weights.h5"
    result_path = r"/home/skm/SKM16/Tmp/green_cover_change/2022-AOI1_fn_rs.tif"
    input_size_green = (128,128,4)
    dil = False
    n_labels = 1
    input_size_green = [128, 128, 4]
    filter_num = [128, 256, 512, 1024]
    from models.models import att_unet
    green_model = att_unet(input_size_green, filter_num, n_labels, stack_num_down=2, stack_num_up=2, activation='ReLU', 
            atten_activation='ReLU', attention='add', output_activation='Sigmoid', batch_norm=False, 
            pool=True, unpool=True, backbone=None, weights='imagenet', freeze_backbone=True,
            freeze_batch_norm=True, name='attunet')

    main_green(image_path, weight_path, result_path, green_model, input_size_green, dil, 
        crop_size=100, batch_size=2, thresh_hold=0.15)
