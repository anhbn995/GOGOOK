import os
import cv2
import gdal

import warnings
warnings.filterwarnings("ignore")

from green_cover.utils import get_img_coords, write_results, padded_for_org_img, \
                                convert_value_obj, predict_green

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

