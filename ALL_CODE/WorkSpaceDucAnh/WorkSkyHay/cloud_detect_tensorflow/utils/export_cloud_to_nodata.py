import os
import glob
import numpy as np
import rasterio
from rasterio.merge import merge
from osgeo import gdalnumeric


def get_list_name_file(path_folder, name_file = '*.tif'):
    """
        Get all file path with file type is name_file.
    """
    list_img_dir = []
    for file_ in glob.glob(os.path.join(path_folder, name_file)):
        head, tail = os.path.split(file_)
        list_img_dir.append(tail)
    return list_img_dir

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
                        nodata=0,
                        ) as dst:
                            dst.write(data)

def get_index_cloud_for_4band(path_mask_cloud):
    """
        get anotation cloud
    """
    src_mask = rasterio.open(path_mask_cloud)
    img_4band = np.empty((4, src_mask.height, src_mask.width))
    for i in range(4):
        img_4band[i] = src_mask.read()#1
    index_cloud = np.where(img_4band != 0)
    return index_cloud

def export_img_cloud_to_nodata(img_path, mask_cloud_path, out_path):
    """
        set cloud is nodata
    """
    # get index_cloud
    index_cloud = get_index_cloud_for_4band(mask_cloud_path)

    # Set nodata
    src = rasterio.open(img_path)
    img = src.read()
    img[index_cloud] = 0
    write_image(img+1, src.height, src.width, src.count, src.crs, src.transform, out_path)

def sort_list_file_by_cloud(dir_predict):
    list_fname = get_list_name_file(dir_predict)
    dict_name = dict.fromkeys(list_fname)
    for fname in list_fname:
        fp = os.path.join(dir_predict, fname)
        raster_file = gdalnumeric.LoadFile(fp)
        count = (raster_file==255).sum()
        dict_name[fname]=count
    dict_name_sort = sorted(dict_name.items(), key=lambda x: x[1])
    list_sort_name = list(dict(dict_name_sort).keys())
    return list_sort_name

def sort_list_file_by_date(list_fp_img_selected):
    list_sort_name = []
    for fp in list_fp_img_selected:
        name = os.path.basename(fp)
        list_sort_name.append(name)
    return list_sort_name

def main_cloud_to_nodata(list_fp_img_selected, dir_predict_float, sort_amount_of_clouds, first_image):
    out_cloud = os.path.join(dir_predict_float, "cloud")
    if not os.path.exists(out_cloud):
        os.makedirs(out_cloud)
    if sort_amount_of_clouds:
        list_fn_sort = sort_list_file_by_cloud(dir_predict_float)
        
    if first_image:
        name = os.path.basename(first_image)
        list_fn_sort.remove(name)
        list_fn_sort.insert(0, name)
        
    for fp in list_fp_img_selected:
        fname = os.path.basename(fp)
        fp_mask = os.path.join(dir_predict_float, fname)
        fp_cloud_rm = os.path.join(out_cloud, fname)
        export_img_cloud_to_nodata(fp, fp_mask, fp_cloud_rm)
    return out_cloud, list_fn_sort


if __name__ == '__main__':
    mask_cloud_path = r"/home/geoai/ml-models/storage/output/f516676cabd94eb2b07fcec390cca25c_mask3.tif"
    # mask_cloud_path = r"/home/geoai/eodata/cloud_detect_tensorflow/cloud_and_shadow/data/out_new_band4.tif"
    img_path = r"/home/geoai/ml-models/storage/example-data/f516676cabd94eb2b07fcec390cca25c.tif"
    out_path =r"/home/geoai/ml-models/storage/output/out.tif"
    export_img_cloud_to_nodata(img_path, mask_cloud_path, out_path)
