import os
import glob
import numpy as np
import multiprocessing

from osgeo import gdal
from functools import partial
from multiprocessing.pool import Pool

def create_list_id(path):
    list_image = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".tif"):
                list_image.append(os.path.join(root, file))
    return list_image
    
def cal_bancut(image_path,num_channel):
    dataset = gdal.Open(image_path)
    band_cut_th = {k: dict(max=0, min=0) for k in range(num_channel)}
    for i_chan in range(num_channel):
        values_ = dataset.GetRasterBand(i_chan+1).ReadAsArray().astype(np.float16)
        values_[values_==0] = np.nan
        band_cut_th[i_chan]['max'] = np.nanpercentile(values_, 98)
        band_cut_th[i_chan]['min'] = np.nanpercentile(values_, 2)
    # print(band_cut_th[i_chan]['max'])
    # print(band_cut_th[i_chan]['min'])
    return band_cut_th

def buil_3_band(image_path, path_create, num_channel):
    list_value = [[1.0, 2177.0],
                 [1.0, 2274.0],
                 [1.0, 2197.0],
                 [1.0, 3570.0]]
    dir_name = os.path.basename(os.path.dirname(image_path))
    image_name = os.path.basename(image_path)[:-4]
    path_out = os.path.join(path_create,dir_name)
    output = os.path.join(path_out,image_name+'.tif')
    band_cut_th = cal_bancut(image_path,num_channel)
    options_list = ['-ot UInt16','-a_nodata 0','-colorinterp_4 undefined']
    for i_chain in range(num_channel):
        options_list.append('-b {}'.format(i_chain+1))
    for i_chain, value in zip(range(num_channel),list_value):
        options_list.append('-scale_{} {} {} {} {} -exponent_{} 1.0'.format(i_chain+1,band_cut_th[i_chain]['min'],band_cut_th[i_chain]['max'],value[0],value[1],i_chain+1))
    options_string = " ".join(options_list)
    gdal.Translate(output,
            image_path,
            options=options_string)
    

    return True

def norm_data(folder_path, num_channel):
    parent = os.path.dirname(folder_path)
    # foder_name = os.path.basename(foder_path)
    core = multiprocessing.cpu_count()//4
    list_id = create_list_id(folder_path)

    path_create = os.path.join(parent.replace(os.path.basename(parent), 'norm_data'))
    if not os.path.exists(path_create):
        os.makedirs(path_create)

    for image_path1 in list_id:
        dir_name = os.path.basename(os.path.dirname(image_path1))
        path_out = os.path.join(path_create,dir_name)
        if not os.path.exists(path_out):
            os.makedirs(path_out)
            
    num_norm_data = len(glob.glob(os.path.join(path_out, '*.tif')))
    num_crop_data = len(list_id)
    if num_norm_data!=num_crop_data:
        p_cnt = Pool(processes=core)
        p_cnt.map(partial(buil_3_band,path_create=path_create,num_channel=num_channel), list_id)
        p_cnt.close()
        p_cnt.join()
    return path_out

if __name__ == "__main__":
    folder_path = '/home/quyet/DATA_ML/Data_work/hanoi_sen2'
    num_channel = 4
    norm_data(folder_path, num_channel)