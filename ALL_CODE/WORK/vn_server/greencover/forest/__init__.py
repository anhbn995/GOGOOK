import os
import sys
import glob
import json
import rasterio
import numpy as np

from forest.convert_tif import write_image
from green_cover.models.models import att_unet
from forest.step_6_predict_forest import inference

def main_only_forest(use_model, weight_path, image_path, result_dir):
    root_dir = sys.path[0]
    config_path = os.path.join(root_dir, 'forest', '%s.json'%(use_model))
    init_model = eval(use_model)
    dict_params = json.load(open(config_path))
    predict_params = dict_params['predict']
    del dict_params['data'], dict_params['strategy'], dict_params['predict']

    model = init_model(**dict_params)
    model.load_weights(weight_path)
    try:
        choose_stage = predict_params['choose_stage']
    except:
        choose_stage = None
    result_path = os.path.join(result_dir, os.path.basename(image_path).replace('.tif', '_forest.tif'))
    inference(model, weight_path, image_path, result_path, predict_params['img_size'], predict_params['crop_size'], 
                                    predict_params['num_band'], predict_params['batch_size'], predict_params['thresh_hold'], 
                                    predict_params['use_dil'], predict_params['rm_small'], choose_stage)
    return result_path
    
    
    
def main_forest(use_model, weight_path, image_path, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    root_dir = sys.path[0]
    config_path = os.path.join(root_dir, 'forest', '%s.json'%(use_model))
    init_model = eval(use_model)
    dict_params = json.load(open(config_path))
    predict_params = dict_params['predict']
    del dict_params['data'], dict_params['strategy'], dict_params['predict']

    model = init_model(**dict_params)
    model.load_weights(weight_path)
    try:
        choose_stage = predict_params['choose_stage']
    except:
        choose_stage = None

    pre_month = ''
    pre_result_path = ''
    for j in image_path:
        result_path = os.path.join(results_path, os.path.basename(j))
        if not os.path.exists(result_path):
            print("Run forest with image path:", j)
            result_path = inference(model, weight_path, j, result_path, predict_params['img_size'], predict_params['crop_size'], 
                                    predict_params['num_band'], predict_params['batch_size'], predict_params['thresh_hold'], 
                                    predict_params['use_dil'], predict_params['rm_small'], choose_stage)
            if not pre_result_path:
                pre_result_path = result_path
                pre_month = j
                pass
            else:
                cur_img = rasterio.open(result_path).read()[0]
                total_pixel_cur = np.unique(cur_img, return_counts=True)[-1][-1]
                # import pdb; pdb.set_trace()
                pre_img = rasterio.open(pre_result_path).read()[0]
                total_pixel_pre = np.unique(pre_img, return_counts=True)[-1][-1]
                if total_pixel_cur > total_pixel_pre *1.07 or total_pixel_cur < total_pixel_pre *0.93:
                    print(pre_result_path, result_path)
                    print("******* BAD *******", (abs(total_pixel_cur-total_pixel_pre)/total_pixel_pre))
                    add_result = inference(model, weight_path, pre_month, result_path, predict_params['img_size'], predict_params['crop_size'], 
                                    predict_params['num_band'], predict_params['batch_size'], 0.05, 
                                    predict_params['use_dil'], predict_params['rm_small'], choose_stage, False)
                    cur_img = np.array(cur_img, np.uint8)
                    add_result = np.array(add_result, np.uint8)
                    cur_img += add_result
                    cur_img[cur_img==2]=1
                    total_pixel_new = np.unique(cur_img, return_counts=True)[-1][-1]
                    print("******* NEW *******", (abs(total_pixel_new-total_pixel_pre)/total_pixel_pre))
                    write_image(j, cur_img)

                else:
                    print("******* GOOD *******")
                pre_result_path = result_path
                pre_month = j
        else:
            pass

    return results_path