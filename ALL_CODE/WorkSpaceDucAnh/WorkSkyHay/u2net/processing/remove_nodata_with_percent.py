import os, glob
import rasterio
import numpy as np
from tqdm import tqdm


def remove_img_nodata(fp_img_mask_check, fp_img, percent_zeros=90):
    with rasterio.open(fp_img_mask_check) as src:
        img = src.read()
        number_pixel = src.height*src.width

    number_pixel_0 = np.count_nonzero(img==0)
    if number_pixel_0/number_pixel > percent_zeros/100:
        if os.path.exists(fp_img_mask_check) and os.path.exists(fp_img):
            os.remove(fp_img_mask_check)
            os.remove(fp_img)
        else:
            print(f"The file img or mask does not exist with name {os.path.basename(fp_img)}")


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

if __name__ == "__main__":
    """Chay 1 cai"""
    # fd_img = r"E:\WorkSpaceSkyMap\Change_detection_Dubai\Data\tach_ra\v2\cut256stride128\B"
    # fd_img_mask_check = r"E:\WorkSpaceSkyMap\Change_detection_Dubai\Data\tach_ra\v2\cut256stride128\labelB"

    # list_name = get_list_name_fp(fd_img_mask_check)

    # i = 0
    # sum_all = len(list_name)
    # phan_tram_giu_lai_mask_rong = 10
    # num_get = sum_all//phan_tram_giu_lai_mask_rong

    # for name in tqdm(list_name):
    #     i += 1
    #     if i%num_get == 0:
    #         continue
    #     else:
    #         fp_img = os.path.join(fd_img, name)
    #         fp_mask = os.path.join(fd_img_mask_check, name)
    #         remove_img_nodata(fp_mask, fp_img, percent_zeros=99.9)


    """Chay nhieu cai"""
    for name in ['train', 'val']:
        fd_img = f"/home/skm/SKM16/Data/IIIII/Data_Train_Pond_fix/Pond_512_fix/{name}/image"
        fd_img_mask_check = f"/home/skm/SKM16/Data/IIIII/Data_Train_Pond_fix/Pond_512_fix/{name}/label"
        # print(fd_img)
        # print(fd_img_mask_check)

        list_name = get_list_name_fp(os.path.normpath(fd_img_mask_check))
        # print(list_name)
        i = 0
        sum_all = len(list_name)
        phan_tram_giu_lai_mask_rong = 10
        num_get = sum_all//phan_tram_giu_lai_mask_rong

        for name in tqdm(list_name):
            i += 1
            if i%num_get == 0:
                continue
            else:
                fp_img = os.path.join(fd_img, name)
                fp_mask = os.path.join(fd_img_mask_check, name)
                remove_img_nodata(fp_mask, fp_img, percent_zeros=99.9)
