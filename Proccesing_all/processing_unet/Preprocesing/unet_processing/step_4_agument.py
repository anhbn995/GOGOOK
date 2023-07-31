import numpy as np
import osr
import time
from osgeo import gdal
import os
import scipy
import scipy.misc
import glob, os
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
core = multiprocessing.cpu_count()
import argparse
import shutil

def save_im(output_path,imgx):
    output = output_path
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(output,imgx.shape[1],imgx.shape[0],(imgx.shape[2]),gdal.GDT_Byte)#gdal.GDT_Byte/GDT_UInt16
    for i in range(1,imgx.shape[2]+1):
       dst_ds.GetRasterBand(i).WriteArray(imgx[:,:,i-1])
       dst_ds.GetRasterBand(i).ComputeStatistics(False)
    dst_ds.FlushCache()

def rotate_angle(org_matrix, angle):
    m1 = np.rot90(org_matrix)
    m2 = np.rot90(m1)
    m3 = np.rot90(m2)
    if angle == "90":
        return m1
    elif angle == "180":
        return m2
    elif angle == "270":
        return m3

def create_list_id(img_dir):
    list_id = []
    os.chdir(img_dir)
    for file in glob.glob("*.tif"):
        list_id.append(file[:-4])
    return list_id


def create_rotate(im_name, img_dir, mask_dir, save_mask_dir, save_img_dir, ouput_mask_path, ouput_img_path):
    rotate = ["90","180","270"]
    ds_img = gdal.Open(os.path.join(img_dir, im_name +".tif"))
    img = ds_img.ReadAsArray()
    img_matrix = np.array(img).swapaxes(0,1).swapaxes(1,2)

    ds_mask = gdal.Open(os.path.join(mask_dir, im_name+".tif"))
    mask = ds_mask.ReadAsArray()

    # im_name = f.split(".")[0]

    for ro in rotate[:]:        
        img = rotate_angle(img_matrix, ro)
        mask1 = rotate_angle(mask, ro)
                        
        save_mask_path = os.path.join(save_mask_dir, ouput_mask_path.format(im_name,ro))
        save_img_path = os.path.join(save_img_dir, ouput_img_path.format(im_name,ro))

        print(save_mask_path)
        print(save_img_path)

        scipy.misc.imsave(save_mask_path, mask1)
        save_im(save_img_path, img)
    
    # del ds_img
    # del ds_mask

def copy_folder_to_folder(source, destination, files):
    for f in files:
        src_path = os.path.join(source,f+".tif")        
        shutil.copy(src_path,destination)
    return

def main(img_dir, mask_dir):
    parent = os.path.dirname(img_dir)
    foder_name = os.path.basename(img_dir)

    save_img_dir = os.path.join(parent,foder_name+"_rotate")
    save_mask_dir = os.path.join(parent,foder_name+"_mask_rotate")

    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    if not os.path.exists(save_mask_dir):
        os.makedirs(save_mask_dir)
  
    ouput_img_path = "{}_{}.tif"
    ouput_mask_path = "{}_{}.tif"

    
    list_id = create_list_id(img_dir)
    p_cnt = Pool(processes=core)
    # im_name, img_dir, mask_dir, save_mask_dir, save_img_dir, ouput_mask_path, ouput_img_path
    result = p_cnt.map(partial(create_rotate
                                ,img_dir=img_dir
                                ,mask_dir=mask_dir
                                ,save_mask_dir=save_mask_dir
                                ,save_img_dir=save_img_dir
                                ,ouput_mask_path=ouput_mask_path
                                ,ouput_img_path=ouput_img_path), list_id)
    p_cnt.close()
    p_cnt.join()

    copy_folder_to_folder(img_dir, save_img_dir, list_id)
    copy_folder_to_folder(mask_dir, save_mask_dir, list_id)
    return save_img_dir, save_mask_dir

if __name__=="__main__":
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
        '--img_dir',
        help='Image directory',
        required=True
    )

    args_parser.add_argument(
        '--mask_dir',
        help='Mask directory',
        required=True
    )

    param = args_parser.parse_args()
    img_dir = param.img_dir
    mask_dir = param.mask_dir
    main(img_dir, mask_dir)