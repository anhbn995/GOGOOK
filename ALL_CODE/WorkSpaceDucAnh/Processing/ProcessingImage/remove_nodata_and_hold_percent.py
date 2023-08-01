import random
import os, glob
import rasterio
import numpy as np
from tqdm import tqdm


def get_list_fn_from_dir_path(folder_dir, type_file = '*.tif'):
        """
            Get all file path with file type is type_file.
        """
        list_fp = []
        for file_ in glob.glob(os.path.join(folder_dir, type_file)):
            head, tail = os.path.split(file_)
            # list_fp.append(os.path.join(head, tail))
            list_fp.append(tail)
        return list_fp

        
def remove_file_from_dir_by_list_fn(dir_path_need_delete, list_fn_need_delete, pbar):
    for fn_need_delete in list_fn_need_delete:
        fp_delete = os.path.join(dir_path_need_delete, fn_need_delete)
        if os.path.exists(fp_delete):
            os.remove(fp_delete)
            pbar.update(1)
        else:
            print(f"The file img or mask does not exist with name {os.path.basename(fp_delete)}")


def check_nodata(fp_img_mask_check, percent_zeros_la_nodata):
    with rasterio.open(fp_img_mask_check) as src:
        img = src.read()
        number_pixel = src.height*src.width
        
    # cho nay la trick cho nhanh hon
    number_pixel_0 = np.count_nonzero(img==0)
    # print(number_pixel_0,'/',number_pixel)
    if number_pixel_0/number_pixel >= percent_zeros_la_nodata/100:
        return True
    else:
        return False


def get_all_fn_nodata_label(dir_mask_label, percent_quyet_dinh_la_nodata, pbar=None):
    list_fn_all = get_list_fn_from_dir_path(dir_mask_label, type_file = '*.tif')
    list_fn_all_zero = []
    for fn in list_fn_all:
        fp_mask = os.path.join(dir_mask_label, fn)
        if check_nodata(fp_mask, percent_quyet_dinh_la_nodata):
            list_fn_all_zero.append(fn)
        if pbar:
            pbar.update(1)
    return list_fn_all_zero, list_fn_all
    

def get_fn_reduce_nodata(all_fn_list, nodata_fn_list, percent_nodata_trong_datatrain, pbar):
    tmp = nodata_fn_list.copy()
    total = len(all_fn_list)
    nodata_sum = len(nodata_fn_list)
    print(total, nodata_sum)
    accept_sum = total - nodata_sum

    if percent_nodata_trong_datatrain < 1:
        percent_nodata_trong_datatrain = percent_nodata_trong_datatrain
    if percent_nodata_trong_datatrain > 1:
        percent_nodata_trong_datatrain = percent_nodata_trong_datatrain/100
        
    # Nếu nodata chiếm < 10% thì không cần bỏ bớt
    if nodata_sum / total < percent_nodata_trong_datatrain:
        return all_fn_list
    i = 0
    # Nếu noda chiếm > 10% thì bỏ bớt
    while nodata_sum / total > percent_nodata_trong_datatrain:
        nodata_fn_list.pop(random.randint(0, len(nodata_fn_list)-1))
        i+=1
        nodata_sum = total - accept_sum - i
    pbar.update(1)
    print(len(list(set(tmp) - set(nodata_fn_list))),'so luong tong')
    return list(set(tmp) - set(nodata_fn_list))


def remove_file_nodata_on_img_and_mask(dir_img, dir_mask_label, percent_nodata_trong_datatrain, percent_quyet_dinh_la_nodata):
    total_tasks = len(glob.glob(os.path.join(dir_mask_label, '*.tif'))) + 1
    with tqdm(total=total_tasks) as pbar: 
        all_nodata_fn_list, all_fn_list = get_all_fn_nodata_label(dir_mask_label, percent_quyet_dinh_la_nodata, pbar)
        print(len(all_nodata_fn_list), len(all_fn_list),"love"*10)
        list_fn_reduce_nodata = get_fn_reduce_nodata(all_fn_list, all_nodata_fn_list, percent_nodata_trong_datatrain, pbar)
    
    print("so thoa man:", len(all_fn_list) - len(all_nodata_fn_list))
    print("so anh nodata xoa", len(list_fn_reduce_nodata))
    print("\n => bat dau xoa")
    total_tasks = 2*len(list_fn_reduce_nodata)
    with tqdm(total=total_tasks) as pbar: 
        remove_file_from_dir_by_list_fn(dir_mask_label, list_fn_reduce_nodata, pbar)
        remove_file_from_dir_by_list_fn(dir_img, list_fn_reduce_nodata, pbar)


if __name__ == "__main__":
    """Chay 1 cai"""
    print('ok')
    dir_img = r'/home/skm/SKM16/A_CAOHOC/Build_data_train_uint8_cua_3Class/Water/gen_cut512stride256_Water_UINT8_512_stride_size_256/image'
    dir_mask_label = r'/home/skm/SKM16/A_CAOHOC/Build_data_train_uint8_cua_3Class/Water/gen_cut512stride256_Water_UINT8_512_stride_size_256/label'
    percent_nodata_trong_datatrain = 10
    percent_zeros_quyet_dinh_la_nodata = 100
    remove_file_nodata_on_img_and_mask(dir_img, dir_mask_label, percent_nodata_trong_datatrain, percent_zeros_quyet_dinh_la_nodata)
    list_fn_all_zero, list_fn_all = get_all_fn_nodata_label(dir_mask_label, percent_quyet_dinh_la_nodata = percent_zeros_quyet_dinh_la_nodata)
    print(f"nike {len(list_fn_all_zero)}/{len(list_fn_all)}")