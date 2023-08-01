import numpy as np
import rasterio


def read_win_and_chuyen_ve0_doc_vung_k_tot(arr, window_size, bo_qua_pixel):
    # # Tạo một mảng numpy 2 chiều kích thước (100, 100) với các giá trị ngẫu nhiên từ 0 đến 255.
    # # arr = np.random.randint(low=0, high=9, size=(20, 16))
    # # arr = np.ones((20, 16))
    # # Thiết lập kích thước cửa sổ và khoảng cách giữa chúng.
    # window_size = 290
    # bo_qua_pixel = 20
    
    stride = window_size + bo_qua_pixel
    # Tạo một mảng boolean để chỉ định các vị trí cần được thay thế bằng giá trị 1.
    mask = np.zeros_like(arr, dtype=bool)

    # for i in range(0, arr.shape[0], stride):
    #     for j in range(0, arr.shape[1], stride):
    #         if i + stride >= arr.shape[0]:
    #             window_size_i = arr.shape[0] - i
    #         else:
    #             window_size_i = window_size
    #         if j + stride >= arr.shape[1]:
    #             window_size_j = arr.shape[1] - j  
    #         else:
    #             window_size_j = window_size
    #         mask[i:i+window_size_i, j:j+window_size_j] = True


    for i in range(0, arr.shape[0] - window_size + 1, stride):
        for j in range(0, arr.shape[1] - window_size + 1, stride):
            mask[i:i+window_size, j:j+window_size] = True


    # Thay thế các giá trị của các cửa sổ 5x5 cách nhau 3 pixel bằng giá trị 1.
    arr[mask] = 0
    return arr




fp_img_predict = r"/home/skm/SKM16/3D/3D/rs5_model_thay/Test_mau_chuan_9_stride300.tif"
out_file = r"/home/skm/SKM16/3D/3D/rs5_model_thay/oke3.tif"
window_size = 290
bo_qua_pixel = 20


with rasterio.open(fp_img_predict) as src:
    meta = src.meta
    arr =  src.read()[0]
    
arr = read_win_and_chuyen_ve0_doc_vung_k_tot(arr, window_size, bo_qua_pixel)




with rasterio.open(out_file, 'w', **meta) as dst:
    dst.write(np.array([arr]))