import argparse
import math
import glob
import scipy
import numpy as np
# import rasterio
import gdal
import h5py
from os import listdir
import os
import sys
import random
import argparse

def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.tif"):
        list_id.append(file[:-4])        
    return list_id

def cal_bancut_band(image_list, img_dir, num_band):
    band_values = {k: [] for k in range(num_band)}
    for im_name in image_list:
        image_path =  os.path.join(img_dir,im_name + ".tif")        
        dataset = gdal.Open(image_path)
        data = dataset.ReadAsArray()
        for i_chan in range(num_band):
            values_ = data[i_chan].ravel().tolist()
            values_ = np.array(
                [v for v in values_ if v != 0]
            )  
            band_values[i_chan].append(values_)
    
    band_cut_th = {k: dict(max=0, min=0) for k in range(num_band)}

    for i_chan in range(num_band):
        band_values[i_chan] = np.concatenate(
            band_values[i_chan]).ravel()
        band_cut_th[i_chan]['max'] = scipy.percentile(
            band_values[i_chan], 98)
        band_cut_th[i_chan]['min'] = scipy.percentile(
            band_values[i_chan], 2)
    
    return band_cut_th

def get_x_default_band(image_list, img_dir, num_band):
    X_val = []
    for im_name in image_list:
        fn = os.path.join(img_dir, im_name + ".tif")        
        dataset = gdal.Open(fn)
        values = dataset.ReadAsArray()[0:num_band]
        for chan_i in range(num_band):
            values[chan_i] = np.clip(values[chan_i], 0, 255)
        X_val.append(values)
    return np.array(X_val)

def get_x_scaled_0_1(image_list, img_dir, num_band, bandcut):
    X_val = []
    for im_name in image_list:
        fn = os.path.join(img_dir, im_name + ".tif")        
        dataset = gdal.Open(fn)
        values = dataset.ReadAsArray()[0:num_band].astype(np.float16)
        for chan_i in range(num_band):
            values[chan_i] = (np.clip(values[chan_i], bandcut[chan_i]['min'], bandcut[chan_i]['max']) - bandcut[chan_i]['min'])/(bandcut[chan_i]['max']-bandcut[chan_i]['min'])
        X_val.append(values)
    return np.array(X_val)    


def get_x(image_list, img_dir, num_band, bandcut):
    X_val = []
    for im_name in image_list:
        fn = os.path.join(img_dir, im_name + ".tif")        
        dataset = gdal.Open(fn)
        values = dataset.ReadAsArray()[0:num_band]
        for chan_i in range(num_band):
            values[chan_i] = np.clip(values[chan_i], bandcut[chan_i]['min'], bandcut[chan_i]['max'])            
        X_val.append(values)
    
    xmean = np.mean(X_val)
    xstd = np.std(X_val)
    xvar = np.var(X_val)

    meta = {
        'mean':xmean,
        'std':xstd,
        'var':xvar
    }

    return np.array(X_val), meta

def get_y(image_list, mask_dir, input_size):
    Y_val = []
    for im_name in image_list:
        fn = os.path.join(mask_dir,im_name +".tif") 
        print(fn)
        dataset = gdal.Open(fn)
        data = dataset.ReadAsArray()
        values = data.astype(np.float32)           
        values = (values > 0.5).astype(np.uint8)
        Y_val.append(values)    
    Y_val = np.array(Y_val).reshape((-1,1,input_size,input_size))    
    return Y_val

def main(img_dir, mask_dir, num_band, input_size, outdir, split, scale=False): 
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    image_list = create_list_id(img_dir)    
    np.random.shuffle(image_list)
    count = len(image_list)
    cut_idx = int(round(count*split))    
    train_list = image_list[0:cut_idx]
    val_list = [id_image for id_image in image_list if id_image not in train_list]

    # bandcut = cal_bancut_band(image_list,img_dir,num_band)
    if scale:
        # bandcut = cal_bancut_band(image_list,img_dir,num_band)
        bandcut={
            0:{
                'min':0.0,
                'max':255
            },
            1:{
                'min':0,
                'max':255
            },
            2:{
                'min':0,
                'max':255
            },
            3:{
                'min':0,
                'max':255
            }
        }
        x_train = get_x_scaled_0_1(train_list,img_dir,num_band, bandcut)
        x_val = get_x_scaled_0_1(val_list, img_dir, num_band, bandcut)
    else:
        x_train = get_x_default_band(train_list,img_dir,num_band)
        x_val = get_x_default_band(val_list, img_dir, num_band)

    y_train = get_y(train_list, mask_dir,input_size)
    y_val = get_y(val_list, mask_dir,input_size)

    # f0 = h5py.File(os.path.join(outdir,'bandcut.hdf5'),'w')
    # f0.create_dataset('bandcut', data=bandcut)
    # f1 = h5py.File(os.path.join(outdir,'x_train_meta.hdf5'),'w')
    # f1.create_dataset('meta', data=meta)
    f2 = h5py.File(os.path.join(outdir,'x_train_usa.hdf5'),'w')
    f2.create_dataset('x_train', data=x_train)
    f3 = h5py.File(os.path.join(outdir,'y_train_usa.hdf5'),'w')
    f3.create_dataset('y_train', data=y_train)
    f4 = h5py.File(os.path.join(outdir,'x_val_usa.hdf5'),'w')
    f4.create_dataset('x_val', data=x_val)
    f5 = h5py.File(os.path.join(outdir,'y_val_usa.hdf5'),'w')
    f5.create_dataset('y_val', data=y_val)    


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
        '--image_dir',
        help='Orginal Image Directory',
        required=True
    )


    args_parser.add_argument(
        '--mask_dir',
        help='Mask directory',
        required=True
    )

    args_parser.add_argument(
        '--input_size',
        help='Net input size',
        required=True
    )

    args_parser.add_argument(
        '--num_band',
        help='number of band use',
        required=True
    )
    
    args_parser.add_argument(
        '--out_dir',
        help='Output folder',
        required=True
    )

    args_parser.add_argument(
        '--split',
        help='train percentage',
        default=0.7,
        type=float
    )

    args_parser.add_argument(
        '--scale',
        help='scale by bandcut',
        default=False,
        type=bool
    )

    param = args_parser.parse_args()
    img_dir= param.image_dir
    mask_dir= param.mask_dir
    input_size = int(param.input_size)
    num_band = int(param.num_band)
    out_dir = param.out_dir
    split = float(param.split)
    scale = param.scale
    
    main(img_dir, mask_dir, num_band, input_size, out_dir, split, scale)