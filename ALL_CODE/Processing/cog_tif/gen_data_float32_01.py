"""
    Generate data for model unint8.
"""

import os
import glob
import gdal
import rasterio
import numpy as np
from tqdm import tqdm



def get_list_name_file(path_folder, name_file = '*.tif'):
    """
        Get all file path with file type is name_file.
    """
    list_img_dir = []
    for file_ in glob.glob(os.path.join(path_folder, name_file)):
        head, tail = os.path.split(file_)
        list_img_dir.append(os.path.join(head,tail))
    return list_img_dir


def get_min_max_image(file_path):
    ds = gdal.Open(file_path,  gdal.GA_ReadOnly)
    numband =  ds.RasterCount
    dict_band_min_max = {1:0}
    for i in range(4):
        print(dict_band_min_max)
        band = ds.GetRasterBand(i + 1)
        min_train, max_train, _, _ = band.GetStatistics(True, True)
        dict_band_min_max.update({ i+1 : {"min": min_train, "max":max_train}})
    return dict_band_min_max, numband


def write_image(data, height, width, numband, crs, tr, out):
    """
        Export numpy array to image by rasterio.
    """
    with rasterio.open(
                        out,
                        'w',
                        driver='GTiff',
                        height=height,
                        width=width,
                        count=numband,
                        dtype=data.dtype,
                        crs=crs,
                        transform=tr,
                        ) as dst:
                            dst.write(data)


def create_img_01(img_path, out_path):
    # get min max tat ca cac band va so band
    dict_min_max_full, numband = get_min_max_image(img_path)
    src = rasterio.open(img_path)
    img_float_01 = np.empty((numband, src.height, src.width))
    for i in range(numband):
        band = src.read(i+1)
        min_tmp = dict_min_max_full[i+1]['min']
        max_tmp = dict_min_max_full[i+1]['max']
        band = np.interp(band, (min_tmp, max_tmp), (0, 1))
        img_float_01[i] = band
    write_image(img_float_01, src.height, src.width, numband, src.crs, src.transform, out_path)


def main(img_dir):
    out_dir = img_dir + "_convert_01_float"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    list_path_file = get_list_name_file(img_dir, name_file = '*.tif')   
    for path_tmp in tqdm(list_path_file):
        name_file = os.path.basename(path_tmp)
        out_file_tmp = os.path.join(out_dir, name_file)
        create_img_01(path_tmp, out_file_tmp)


img_path = r"/home/skm/SKM/WORK/KhongGianXanh/1_IMG_ORGIN/Create_data_train/img_orgin_COG"
main(img_path)