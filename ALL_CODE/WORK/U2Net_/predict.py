import rasterio
import math
import tensorflow as tf
import os
import pandas as pd
import csv
import time
import geopandas as gp
from component_reader import DaReader, DaWriterKernel, DaStretchKernel, DaUnetPredictKernel, DaSyntheticKernel, MorphologyKernel
from rio_tiler.io import COGReader
from datetime import date
from models.import_module import DexiNed, Model_U2Netp, Model_U2Net, Adalsn, Model_UNet3plus
import glob
from tqdm import tqdm
from rasterio.windows import Window
from matplotlib.patches import Polygon
import rasterio.features
from shapely.geometry import Polygon, mapping
from rasterio.crs import CRS
import numpy as np
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))

def get_quantile_schema(img):
    qt_scheme = []
    with COGReader(img) as cog:
        stats = cog.statistics()
        for _, value in stats.items():
            qt_scheme.append({
                'p2': value['percentile_2'],
                'p98': value['percentile_98'],
            })
    return qt_scheme
    


def predict_farm(image_path, output, model1, model2=None, model3=None, pbars=None):
    quantile_schema = get_quantile_schema(image_path)

    with DaReader(image_path) as img:
        profile = img.profile
        profile['compress'] = 'DEFLATE'
        profile['nodata'] = 0
        profile['count'] = 1

        size = 480 
        padding = size//8
        size_predict = size - size//4
        stretch_kernel = DaStretchKernel(quantile_schema, profile)
        predictor_kernel = DaUnetPredictKernel(model1, model2, model3, size, padding)
        morphology_kernel = MorphologyKernel()
        writer_kernel = DaWriterKernel(output, **profile)
        synthetic = DaSyntheticKernel([stretch_kernel, predictor_kernel, morphology_kernel, writer_kernel])
        img.multi_execute([synthetic], size=size_predict, buffer=padding, pbars=pbars)

if __name__ == '__main__':
#     model_path = './model/dexined_farm_v2.h5'
#     model = DexiNed(480, 3)
#     model.load_weights(model_path)

    # model_path1 = './model/u2net_farm_v3.h5'
    # model1 = Model_U2Net(480, 3)
    # model1.load_weights(model_path1)
    
    # model_path2 = './model/Adalsn.h5'
    model_path2 = './model/adalsn_farm_v3.h5'
    model2 = NetWork(480, 3)
    model2.load_weights(model_path2)
    
#     model_path3 = './model/UNet3plus_v2.h5'
#     model3 = Model_UNet3plus(480, 3)
#     model3.load_weights(model_path3)
    
    # folder = "/mnt/data/public/farm-bing18/Farm_india_3village/"
    out_folder = '/mnt/data/public/farm-bing18/Farm_india_3village/predict/mask_adalsn/'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
        
    # name_image = os.listdir(folder)
    name_image = glob.glob('/mnt/data/public/farm-bing18/3Band_zoom18/*/*_gmap.tif')
    with tqdm(total=len(name_image), ncols=77) as pbar:
        for _file in name_image:
            try:
                # input_path = folder + _file
                # output_path = out_folder + _file
                input_path = _file
                output_path = _file.replace('_gmap.tif','_gmap_mask.tif')
                if not os.path.exists(output_path):
                    predict_farm(input_path, output_path, model2, pbars=pbar)
            except Exception as e:
                print(e)
            pbar.update(1)

#     input_path = '/media/skymap/Learnning/public/farm-bing18/Bingmaps_wajo/tile_z11_1706_999.tif'
    # input_path = '/mnt/data/public/farm-bing18/sandip_farm/bing.tif'
    # output_path = '/mnt/data/public/farm-bing18/sandip_farm/test.tif'
    # predict_farm(input_path, output_path, model1)