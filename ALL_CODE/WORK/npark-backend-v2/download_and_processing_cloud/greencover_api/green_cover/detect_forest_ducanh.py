import os
import sys
import glob
import json
import rasterio
import numpy as np
from osgeo import gdal, osr, ogr

"""Chay rieng cho dectet_green"""
import cv2
import rasterio
import numpy as np

from tqdm import tqdm

import cv2
import rasterio
import numpy as np
import skimage.morphology
from models.models import att_unet

def predict(model, values, img_coords, num_band, h, w, padding, crop_size, 
            input_size, batch_size, thresh_hold, choose_stage):
    cut_imgs = []
    for i in range(len(img_coords)):
        im = get_im_by_coord(values, img_coords[i][0], img_coords[i][1],
                            num_band,padding, crop_size, input_size)
        cut_imgs.append(im)

    a = list(range(0, len(cut_imgs), batch_size))

    if a[len(a)-1] != len(cut_imgs):
        a[len(a)-1] = len(cut_imgs)

    y_pred = []
    for i in tqdm(range(len(a)-1)):
        x_batch = []
        x_batch = np.array(cut_imgs[a[i]:a[i+1]])
        y_batch = model.predict(x_batch)
        if len(model.outputs)>1:
            y_batch = y_batch[choose_stage]
        y_pred.extend(y_batch)
    big_mask = np.zeros((h, w)).astype(np.float16)
    for i in range(len(cut_imgs)):
        true_mask = y_pred[i].reshape((input_size,input_size))
        true_mask = (true_mask>thresh_hold).astype(np.uint8)
        true_mask = (cv2.resize(true_mask,(input_size, input_size), interpolation = cv2.INTER_CUBIC)>thresh_hold).astype(np.uint8)
        start_x = img_coords[i][1]
        start_y = img_coords[i][0]
        big_mask[start_x-padding:start_x-padding+crop_size, start_y-padding:start_y -
                    padding+crop_size] = true_mask[padding:padding+crop_size, padding:padding+crop_size]
    del cut_imgs
    return big_mask


def dilation_obj(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask_base = cv2.dilate(img,kernel,iterations = 1)
    return mask_base

def remove_small_items(img, threshhlod_rm_holes= 256, threshhold_rm_obj=100):
    img_tmp = np.asarray(img, dtype=np.bool)
    img_tmp = skimage.morphology.remove_small_holes(img_tmp, area_threshold=threshhlod_rm_holes)
    img_tmp = skimage.morphology.remove_small_objects(img_tmp, min_size=threshhold_rm_obj)
    return img_tmp

def write_image(image_path, result_path, img):
    with rasterio.open(image_path) as src:
        out_meta = src.meta
    print("Write image...")

    with rasterio.Env():
        # Write an array as a raster band to a new 8-bit file. For
        # the new file's profile, we start with the profile of the source
        profile = out_meta

        # And then change the band count to 1, set the
        # dtype to uint8, and specify LZW compression.
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw')

    with rasterio.open(result_path, 'w', **profile) as dst:
        dst.write(img.astype(np.uint8),1)
    return result_path



def convert_value_obj(big_mask):
    mask_base = big_mask.astype(np.uint8)
    mask_base[mask_base==0]=2
    mask_base[mask_base==1]=0
    mask_base[mask_base==2]=1
    return mask_base

def get_im_by_coord(org_im, start_x, start_y,num_band, padding, crop_size, input_size):
    startx = start_x-padding
    endx = start_x+crop_size+padding
    starty = start_y - padding
    endy = start_y+crop_size+padding
    result=[]
    img = org_im[starty:endy, startx:endx]
    img = img.swapaxes(2,1).swapaxes(1,0)
    for chan_i in range(num_band):
        result.append(cv2.resize(img[chan_i],(input_size, input_size), interpolation = cv2.INTER_CUBIC))
    return np.array(result).swapaxes(0,1).swapaxes(1,2)

def get_img_coords(w, h, padding, crop_size):
    new_w = w + 2*padding
    new_h = h + 2*padding
    cut_w = list(range(padding, new_w - padding, crop_size))
    cut_h = list(range(padding, new_h - padding, crop_size))

    list_hight = []
    list_weight = []
    for i in cut_h:
        if i < new_h - padding - crop_size:
            list_hight.append(i)
    list_hight.append(new_h-crop_size-padding)

    for i in cut_w:
        if i < new_w - crop_size - padding:
            list_weight.append(i)
    list_weight.append(new_w-crop_size-padding)

    img_coords = []
    for i in list_weight:
        for j in list_hight:
            img_coords.append([i, j])
    return img_coords

def padded_for_org_img(values, num_band, padding):
    padded_org_im = []
    for i in range(num_band):
        band = np.pad(values[i], padding, mode='reflect')
        padded_org_im.append(band)

    values = np.array(padded_org_im).swapaxes(0,1).swapaxes(1,2)
    print(values.shape)
    del padded_org_im
    return values

def predict_green(model, values, img_coords, num_band, h, w, padding, crop_size, 
            input_size, batch_size, thresh_hold):
    cut_imgs = []
    for i in range(len(img_coords)):
        im = get_im_by_coord(values, img_coords[i][0], img_coords[i][1],
                            num_band,padding, crop_size, input_size)
        cut_imgs.append(im)

    a = list(range(0, len(cut_imgs), batch_size))

    if a[len(a)-1] != len(cut_imgs):
        a[len(a)-1] = len(cut_imgs)

    y_pred = []
    for i in tqdm(range(len(a)-1)):
        x_batch = []
        x_batch = np.array(cut_imgs[a[i]:a[i+1]])
        y_batch = model.predict(x_batch)
        y_pred.extend(y_batch)
    big_mask = np.zeros((h, w)).astype(np.float16)
    for i in range(len(cut_imgs)):
        true_mask = y_pred[i].reshape((input_size,input_size))
        true_mask = (true_mask>thresh_hold).astype(np.uint8)
        true_mask = (cv2.resize(true_mask,(input_size, input_size), interpolation = cv2.INTER_CUBIC)>thresh_hold).astype(np.uint8)
        start_x = img_coords[i][1]
        start_y = img_coords[i][0]
        big_mask[start_x-padding:start_x-padding+crop_size, start_y-padding:start_y -
                    padding+crop_size] = true_mask[padding:padding+crop_size, padding:padding+crop_size]
    del cut_imgs
    return big_mask

def predict_water(model, values, img_coords, num_band, h, w, padding, crop_size, 
            input_size, batch_size, thresh_hold):
    cut_imgs = []
    for i in range(len(img_coords)):
        im = get_im_by_coord(values, img_coords[i][0], img_coords[i][1],
                            num_band,padding, crop_size, input_size)
        cut_imgs.append(im)

    a = list(range(0, len(cut_imgs), batch_size))

    if a[len(a)-1] != len(cut_imgs):
        a[len(a)-1] = len(cut_imgs)

    y_pred = []
    for i in tqdm(range(len(a)-1)):
        x_batch = []
        x_batch = np.array(cut_imgs[a[i]:a[i+1]])
        y_batch = model.predict(x_batch)
        y_batch = y_batch[-1]
        y_pred.extend(y_batch)
    big_mask = np.zeros((h, w)).astype(np.float16)
    for i in range(len(cut_imgs)):
        true_mask = y_pred[i].reshape((input_size,input_size))
        true_mask = (true_mask>thresh_hold).astype(np.uint8)
        true_mask = (cv2.resize(true_mask,(input_size, input_size), interpolation = cv2.INTER_CUBIC)>thresh_hold).astype(np.uint8)
        start_x = img_coords[i][1]
        start_y = img_coords[i][0]
        big_mask[start_x-padding:start_x-padding+crop_size, start_y-padding:start_y -
                    padding+crop_size] = true_mask[padding:padding+crop_size, padding:padding+crop_size]
    del cut_imgs
    return big_mask

def write_results(mask_base, image_path, result_path):
    with rasterio.open(image_path) as src:
        transform1 = src.transform
        w,h = src.width,src.height
        crs = src.crs
    new_dataset = rasterio.open(result_path, 'w', driver='GTiff',
                                height = h, width = w,
                                count=1, dtype="uint8",
                                crs=crs,
                                transform=transform1,
                                compress='lzw')
    new_dataset.write(mask_base,1)
    new_dataset.close()



def inference(model, weight_path, image_path, result_path, img_size=128, crop_size=100, num_band=4,
            batch_size=2, thresh_hold=0.5, dil=False, rm_small= False, choose_stage=None, return_img=True):
    infer_model = model.load_weights(weight_path) 

    dataset = gdal.Open(image_path)
    values = dataset.ReadAsArray()[0:num_band]/255
    h,w = values.shape[1:3]    
    padding = int((img_size - crop_size)/2)
    img_coords = get_img_coords(w, h, padding, crop_size)
    values = padded_for_org_img(values, num_band, padding)
    big_mask = predict(model, values, img_coords, num_band, h, w, padding, crop_size, 
                        img_size, batch_size, thresh_hold, choose_stage)
    
    # if dil:
    #     big_mask = dilation_obj(big_mask)
    
    # if rm_small:
    #     big_mask = remove_small_items(big_mask, threshhlod_rm_holes=256,
    #                                     threshhold_rm_obj=100)
    
    # big_mask[big_mask==0]=2
    # big_mask[big_mask==1]=0
    # big_mask[big_mask==2]=1
    if return_img:
        result_path = write_image(image_path, result_path, big_mask)
        return result_path
    else:
        return big_mask





def main_forest(use_model, weight_path, image_path, results_path):
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    root_dir = sys.path[0]
    config_path = os.path.join(root_dir, 'download_and_processing_cloud','greencover_api','forest', '%s.json'%(use_model))
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
                    write_image(j, cur_img, add_result)

                else:
                    print("******* GOOD *******")
                pre_result_path = result_path
                pre_month = j
        else:
            pass

    return results_path

"""Chay rieng cho dectet_green"""
def main_tmp(use_model, weight_path, image_path, results_path):

    config_path = r"/home/skm/SKM16/IMAGE/npark/npark-backend-v2/download_and_processing_cloud/greencover_api/forest/att_unet.json"
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
    result_path = inference(model, weight_path, image_path, results_path, predict_params['img_size'], predict_params['crop_size'], 
                                    predict_params['num_band'], predict_params['batch_size'], predict_params['thresh_hold'], 
                                    predict_params['use_dil'], predict_params['rm_small'], choose_stage)
if __name__=="__main__":
    use_model = 'att_unet'
    image_path = r"/home/skm/SKM16/Tmp/green_cover_change/2019-AOI1_fn.tif"
    weight_path = r"/home/skm/public_mount/DucAnhtmp/weights/forest_weights_v2.h5"
    result_path = r"/home/skm/SKM16/Tmp/green_cover_change/2019-AOI1_fn_rs_forest.tif"
    main_tmp(use_model, weight_path, image_path, result_path)