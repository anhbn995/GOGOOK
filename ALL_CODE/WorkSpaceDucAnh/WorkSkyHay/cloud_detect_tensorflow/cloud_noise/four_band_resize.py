# -*- coding: utf-8 -*-
import glob, os
from osgeo import gdal, gdalconst, ogr, osr

def resize(image_path,path_create, resolution_origin, resolution_destination):
    image_id =  os.path.basename(image_path)
    # output = os.path.join(path_create,image_id)
    output=path_create
    ds = gdal.Open(image_path)
    # resolution_origin = 0.1
    # resolution_destination = 0.2
    size_destination = resolution_origin/resolution_destination*100
    print(resolution_destination/resolution_origin)
    options_list = [
    f'-outsize {size_destination}% {size_destination}%',
    '-of GTiff',
    '-r cubic',
    '-ot Byte'
    ] 
    options_string = " ".join(options_list)

    gdal.Translate(output,
                image_path,
                options=options_string)
    return output

image_list = glob.glob(r"/home/geoai/eodata/cloud_detect_tensorflow/raw_final/noise_data/*.tif")
out_path = r"/home/geoai/eodata/cloud_detect_tensorflow/raw_final/noise_data/"
resolution_origin = 0.05
resolution_destination = 0.1

# main
for img_path in image_list:
    resize(image_path, out_path, resolution_origin, resolution_destination)