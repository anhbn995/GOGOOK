import os
import shutil
import rasterio
from osgeo import gdal
from app.utils.path import make_temp_folder

def reproject_image(src_path, dst_path, dst_crs='EPSG:4326'):
    with rasterio.open(src_path) as ds:
        nodata = ds.nodata or 0
    if ds.crs.to_string() != dst_crs:
        print(f'convert to {dst_crs}')
        temp_path = dst_path.replace('.tif', 'temp.tif')
        option = gdal.TranslateOptions(gdal.ParseCommandLine("-co \"TFW=YES\""))
        gdal.Translate(temp_path, src_path, options=option)
        option = gdal.WarpOptions(gdal.ParseCommandLine("-t_srs {} -dstnodata {}".format(dst_crs, nodata)))
        gdal.Warp(dst_path, temp_path, options=option)
        os.remove(temp_path)
    else:
        print(f'coppy image to {dst_path}')
        shutil.copyfile(src_path, dst_path)
        print('done coppy')



def get_pixel_size(image_path):
    temp_folder = make_temp_folder()
    temp_reproject = f'{temp_folder}/3857_mosaic.tif'
    reproject_image(image_path, temp_reproject, 'EPSG:3857')
    with rasterio.open(temp_reproject) as dt:
        x_size = dt.res[0]
        y_size = dt.res[1]
    shutil.rmtree(temp_folder)
    return x_size, y_size