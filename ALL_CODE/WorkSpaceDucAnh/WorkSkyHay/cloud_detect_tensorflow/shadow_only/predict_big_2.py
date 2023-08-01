import os
import numpy as np
import tensorflow as tf
import os
import tensorflow as tf
from tensorflow.keras import layers, backend, Model, utils
import rasterio
import cv2
from tensorflow.compat.v1.keras.backend import set_session
from rasterio.features import sieve

from tqdm import *
import glob

#chinh lai detect trong class unet
import rasterio
import numpy as np
#import model
import os
import cv2
from tqdm import *
#####################################    


def predict(img_url, threshold = 0.5, cnn_model =None):
    num_band = 4
    batch_size = 2
    input_size = 512
    INPUT_SIZE = 512
    crop_size = 400
    #cnn_model = model.unet_basic_2(num_channel=num_band, size=INPUT_SIZE)
    #cnn_model.load_weights(FN_MODEL)  
    with rasterio.open(img_url) as src:
        values = src.read()[0:num_band]
        msk = src.read_masks(1)
    
    
    # print("values",len(values))
    h,w = values.shape[1:3]    
    padding = int((input_size - crop_size)/2)
    # print("pading",padding)
    padded_org_im = []
    cut_imgs = []
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
    
    for i in range(num_band):
        band = np.pad(values[i], padding, mode='reflect')
        padded_org_im.append(band)


    values = np.array(padded_org_im).swapaxes(0,1).swapaxes(1,2)
    del padded_org_im

    def get_im_by_coord(org_im, start_x, start_y,num_band):
        startx = start_x-padding
        endx = start_x+crop_size+padding
        starty = start_y - padding
        endy = start_y+crop_size+padding

        result=[]
        img = org_im[starty:endy, startx:endx]
        img = img.swapaxes(2,1).swapaxes(1,0)
        for chan_i in range(num_band):
            result.append(cv2.resize(img[chan_i],(INPUT_SIZE, INPUT_SIZE), interpolation = cv2.INTER_CUBIC))
        return np.array(result).swapaxes(0,1).swapaxes(1,2)

    for i in range(len(img_coords)):
        im = get_im_by_coord(
            values, img_coords[i][0], img_coords[i][1],num_band)
        cut_imgs.append(im)


    a = list(range(0, len(cut_imgs), batch_size))

    if a[len(a)-1] != len(cut_imgs):
        a[len(a)-1] = len(cut_imgs)

    y_pred = []

    for i in range(len(a)-1):
        x_batch = []
        x_batch = np.array(cut_imgs[a[i]:a[i+1]]).astype(np.float16)/np.finfo(np.float16).max


        y_batch = cnn_model.predict(x_batch)
        y_pred.extend(y_batch)
    big_mask = np.zeros((h, w))
    for i in tqdm(range(len(cut_imgs))):
        if np.count_nonzero(cut_imgs[i]) > 0:
            true_mask = y_pred[i].reshape((INPUT_SIZE,INPUT_SIZE))
            true_mask = (true_mask>0.5).astype(np.uint8)
        else:
            true_mask = np.zeros((INPUT_SIZE,INPUT_SIZE))
        true_mask = (cv2.resize(true_mask,(input_size, input_size), interpolation = cv2.INTER_CUBIC)>0.5).astype(np.uint8)
        start_x = img_coords[i][1]
        start_y = img_coords[i][0]

        big_mask[start_x-padding:start_x-padding+crop_size, start_y-padding:start_y -
                    padding+crop_size] = true_mask[padding:padding+crop_size, padding:padding+crop_size]

    del cut_imgs
    big_mask = (big_mask > 0.5).astype(np.uint8)
    return big_mask*255


def predict_all_2(base_path,outputFileName, cnn_model):

    mask_base = predict(base_path, threshold = 0.5, cnn_model =cnn_model)
    print(mask_base.shape, "zz")
    with rasterio.open(base_path) as src:
        transform1 = src.transform
        w,h = src.width,src.height
        crs = src.crs

    new_dataset = rasterio.open(outputFileName, 'w', driver='GTiff',
                                height = h, width = w,
                                count=1, dtype="uint8",
                                crs=crs,
                                transform=transform1,
                                compress='lzw')

    new_dataset.write(mask_base,1)
    new_dataset.close()

   

if __name__ == '__main__':
    img_data = r"/home/geoai/eodata/cloud_detect_tensorflow/raw_final/predict/232b3e77828845b49d76ab1a83b765ef.tif"
    # img_data = r"/home/geoai/eodata/cloud_detect_tensorflow/raw_final/test2.tif"
    output_im = img_data.replace(".tif", "_label.tif")
    # output_im = r"/home/geoai/eodata/cloud_detect_tensorflow/raw_final/232b3e77828845b49d76ab1a83b765ef_label.tif"
    FN_MODEL = r"./cp.h5"
    from model import build_model
    import logging
    unet = build_model((None,None,4), 46)
    logging.info("Loading model {}".format(unet.name))
    unet.load_weights(FN_MODEL)
    logging.info("Model loaded !")
    
    predict_all_2(img_data, output_im, cnn_model=unet)
    print('Process successfully, output saved at', output_im)