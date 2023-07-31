import rasterio
import os, glob
import numpy as np
from tqdm import tqdm

def remove_img_nodata(fp_img_mask_check, fp_img, percent_zeros=90):
    with rasterio.open(fp_img_mask_check) as src:
        img = src.read()
        number_pixel = src.height*src.width

    number_pixel_0 = np.count_nonzero(img==0)
    if number_pixel_0/number_pixel > percent_zeros/100:
        if os.path.exists(fp_img_mask_check):
            os.remove(fp_img_mask_check)
        else:
            print("The file mask does not exist")

        if os.path.exists(fp_img):
            os.remove(fp_img)
        else:
            print("The file img does not exist")


# def remove_img_nodata(fp_img_mask_check, fp_img, percent_zeros=90):
#     with rasterio.open(fp_img) as src:
#         img = src.read()
#         number_pixel = src.height*src.width

    
#     if np.all(img == 0):
#         if os.path.exists(fp_img_mask_check):
#             os.remove(fp_img_mask_check)
#         else:
#             print("The file mask does not exist")

#         if os.path.exists(fp_img):
#             os.remove(fp_img)
#         else:
#             print("The file img does not exist")


def get_list_name_fp(folder_dir, type_file = '*.tif'):
        """
            Get all file path with file type is type_file.
        """
        list_fp = []
        for file_ in glob.glob(os.path.join(folder_dir, type_file)):
            head, tail = os.path.split(file_)
            # list_fp.append(os.path.join(head, tail))
            list_fp.append(tail)
        return list_fp

fd_img = r"/home/skm/SKM16/Data/ThaiLandChangeDetection/BD_Chang/Data_Train_and_Model/cut256_128/image_cut_img_crop"
fd_img_mask_check = r"/home/skm/SKM16/Data/ThaiLandChangeDetection/BD_Chang/Data_Train_and_Model/cut256_128/image_cut_img_mask_crop"

list_name = get_list_name_fp(fd_img_mask_check)

i = 0
sum_all = len(list_name)
phan_tram = 10
num_get = sum_all//phan_tram
for name in tqdm(list_name):
    i += 1
    if i%num_get == 0:
        continue
    else:
        fp_img = os.path.join(fd_img, name)
        fp_mask = os.path.join(fd_img_mask_check, name)
        # print(fp_img)
        remove_img_nodata(fp_mask, fp_img, percent_zeros=99.9)
