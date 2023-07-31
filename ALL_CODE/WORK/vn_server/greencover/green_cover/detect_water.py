import os
import gdal
import warnings
warnings.filterwarnings("ignore")

from green_cover.utils import get_img_coords, write_results, padded_for_org_img, \
                                convert_value_obj, predict_water

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

