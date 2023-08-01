import sys
import os, glob
import rasterio
import numpy as np
from tqdm import tqdm

sys.path.append(r'E:\WorkSpaceDucAnh\Statistic')
from morphology import *


def rm_by_value(img, value_xoa, area_pixel_remove):
    index_mat = np.where(img == value_xoa)
    img_mat = np.zeros_like(img)
    img_mat[index_mat] = 255
    # đây là xóa đóm
    new_img = remove_area_small(img_mat[0], area_pixel_remove, value_draw=255)
    # đây là xóa lỗ
    new_img = remove_area_small(255-new_img, area_pixel_remove, value_draw=255)
    new_img = 255-new_img
    index_value_xoa = np.where(np.array([new_img]) != 0)
    return index_value_xoa


def xoa_vung_nho_cua_change(dir_out, dir_in, area_pixel_remove, value_mat=None, value_them=None):
    os.makedirs(dir_out, exist_ok=True)
    list_fp = glob.glob(os.path.join(dir_in, '*.tif'))
    for fp in list_fp:
        if value_mat or value_them:
            fp_out = os.path.join(dir_out, os.path.basename(fp))
            with rasterio.open(fp) as src:
                meta = src.meta
                img  = src.read()

            img_rs = np.zeros_like(img)
            if value_mat:
                index_mat = rm_by_value(img, value_mat, area_pixel_remove)
                img_rs[index_mat] = value_mat
            if value_them:
                index_them = rm_by_value(img, value_them, area_pixel_remove)
                img_rs[index_them] = value_them
            print(np.unique((img_rs)))
            with rasterio.open(fp_out, 'w', **meta) as dst:
                dst.write(img_rs)
                dst.write_colormap(1, {
                        0: (0,0,0, 0), 
                        1: (255,0,0,0),
                        2: (31,255,15,0)
                        })


if __name__=='__main__':
    dir_in = r'E:\TMP_XOA\Forest_tiep\rs_change_raster'
    area_pixel_remove = 4
    dir_out = os.path.join(dir_in, f'morphology_remove_{area_pixel_remove}')
    value_mat = 1
    value_them = 2
    xoa_vung_nho_cua_change(dir_out, dir_in, area_pixel_remove, value_mat=1, value_them=2)
                
            

                




        




# list_fp = glob.glob(r'E:\TMP_XOA\Data_same_size\2022\*.tif')
# out_dir = r'E:\TMP_XOA\Data_same_size_mopho\2022'
# os.makedirs(out_dir, exist_ok=True)
# for fp in tqdm(list_fp):
#     with rasterio.open(fp) as src:
#         meta = src.meta
#         img  = src.read()

#     img[img!=0]=255
#     # img[img==2]=0
#     # new_img = closing(img[0],  size_kernel=3,  shape_kernel="rec")
#     new_img = remove_area_small(img[0], 4, value_draw=255)
#     new_img = remove_area_small(255-new_img, 2, value_draw=255)
#     new_img = 255-new_img
#     fp_out = os.path.join(out_dir, os.path.basename(fp))
#     with rasterio.open(fp_out, 'w', **meta) as dst:
#         dst.write(np.array([new_img]))