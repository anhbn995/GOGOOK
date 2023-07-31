import os
import sys
import json
import argparse

from models.core import *
from scripts.predict import inference

def main(use_model, weight_path, image_path):
    root_dir = sys.path[0]
    config_path = os.path.join(root_dir, 'configs', '%s.json'%(use_model))
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

    result_path, big_mask = inference(model, weight_path, image_path, predict_params['img_size'], predict_params['crop_size'], 
                            predict_params['num_band'], predict_params['batch_size'], predict_params['thresh_hold'], 
                            predict_params['use_dil'], predict_params['rm_small'], choose_stage)
    return result_path, big_mask


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_model', type=str, default='att_unet', help='Choose model to use', required=False)
    parser.add_argument('--img_path', type=str, default='/home/quyet/DATA_ML/Projects/Green Cover Npark Singapore/results/T1-2022/DA/T1-2022_DA.tif', 
                        help='Define path of folder contains images', required=False)
    parser.add_argument('--weight_path', type=str, default='/home/quyet/DATA_ML/Projects/segmentation/weights/att_unet/att_unet_forest_256_1class_train.h5', 
                        help='Define path of weight', required=False)

    args = parser.parse_args()
    main(args.name_model, args.weight_path, args.img_path)