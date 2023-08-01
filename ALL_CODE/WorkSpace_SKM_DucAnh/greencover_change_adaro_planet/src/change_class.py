import cv2
import rasterio
import numpy as np
from tqdm import tqdm


def transform_from_truoc_to_sau_one_window(arr_last, arr_next, list_value_class_truoc=[255], list_value_class_sau=[255], value_chuyen_doi=1):
    """ Run object detection change for each part same shape.

    Args:
        arr_last (numpy array): last month's classification mask
        arr_next (numpy array): next month's classification mask
        value_class_truoc (int, optional): value object last month. Defaults to 255.
        value_class_sau (int, optional): value object next month. Defaults to 255.
        value_mat (int, optional): value green to non. Defaults to 1.
        value_them (int, optional): value non to green. Defaults to 2.

    Returns:
        numpy array: change green mask
    """
    
    idx_cloud_truoc = np.where(arr_last == 5)
    idx_cloud_sau = np.where(arr_next == 5)
    
    # print(list_value_class_truoc, list_value_class_sau)
    # print(np.unique(arr_last),1)
    condition_truoc = " | ".join(["(arr_last == {})".format(value) for value in list_value_class_truoc])
    arr_truoc = np.where(eval(condition_truoc), 1, 0)
    # print(np.unique(arr_truoc))
    
    # print(np.unique(arr_next), 1)
    condition_sau = " | ".join(["(arr_next == {})".format(value) for value in list_value_class_sau])
    arr_sau = np.where(eval(condition_sau), 1, 0)
    # print(np.unique(arr_sau))
    
    mask_chuyen = arr_truoc + arr_sau
    index_chuyen = np.where(mask_chuyen == 2)
    mask_change = np.zeros_like(arr_last, dtype='uint8')
    mask_change[index_chuyen] = value_chuyen_doi
    mask_change[idx_cloud_truoc] = 0
    mask_change[idx_cloud_sau] = 0
    # print(np.unique(mask_chuyen))
    return mask_change


def main_change_green(fp_img_truoc, fp_img_sau, value_class_truoc=[255], value_class_sau=[255], value_change=1, area_maximum = 550):
    """_summary_

    Args:
        fp_change_out (_type_): _description_
        fp_img_truoc (_type_): _description_
        fp_img_sau (_type_): _description_
        value_class_truoc (list, optional): _description_. Defaults to [255].
        value_class_sau (list, optional): _description_. Defaults to [255].
        value_change (int, optional): _description_. Defaults to 1.
    """
    

    # try:
    if True:
        window_size = 500000
        with rasterio.open(fp_img_truoc) as src_truoc:
            meta = src_truoc.meta
            shape = src_truoc.shape
            with rasterio.open(fp_img_sau) as src_sau:
                if shape[1] < window_size:
                    # mask_change_win = find_change_2_mask_different_one_window(src_truoc.read(), src_sau.read(), value_class_truoc=value_class_truoc,\
                    #                                                             value_class_sau=value_class_sau, value_mat=value_mat, value_them=value_them)
                    mask_change_win = transform_from_truoc_to_sau_one_window(src_truoc.read(), src_sau.read(), list_value_class_truoc=value_class_truoc, list_value_class_sau=value_class_sau, value_chuyen_doi=value_change)
                else:
                    mask_change_win = np.empty(shape, rasterio.uint8)
                    num_windows_height = (src_truoc.height + window_size - 1) // window_size
                    num_windows_width = (src_truoc.width + window_size - 1) // window_size
                
                    for row in range(num_windows_height):
                        for col in range(num_windows_width):
                            # Tính toán vị trí của khung hình
                            window_row_start = row * window_size
                            window_row_end = min((row + 1) * window_size, src_truoc.height)
                            window_col_start = col * window_size
                            window_col_end = min((col + 1) * window_size, src_truoc.width)
                            
                            window = ((window_row_start, window_row_end), (window_col_start, window_col_end))
                            arr_last = src_truoc.read(window=window)
                            arr_next = src_sau.read(window=window)
                            
                            # mask_change_win = find_change_2_mask_different_one_window(arr_last, arr_next, value_class_truoc=value_class_truoc,\
                            #                                                     value_class_sau=value_class_sau, value_mat=value_mat, value_them=value_them)
                            mask_change_win = transform_from_truoc_to_sau_one_window(arr_last, arr_next, list_value_class_truoc=value_class_truoc, list_value_class_sau=value_class_sau, value_chuyen_doi=value_change)
                            
                            mask_change_win[window_row_start:window_row_end, window_col_start:window_col_end] = mask_change_win
                    mask_change_win = np.array([mask_change_win])
    
        # with rasterio.open(fp_change_out, 'w', **meta) as dst:
        #     dst.write(mask_change_win)
        #     # dst.update_tags(AREA_OR_POINT="green='1'")
        #     # dst.write_colormap(1, {
        #     #         0: (0,0,0, 0), 
        #     #         1: (31,255,15,0),
        #     #         2: (255,0,0,0)
        #     #         })
        #     dst.write_colormap(1, {
        #             # 0: (0,0,0, 0), 
        #             value_change: (255,0,0,0)
        #             # 2: (255,0,0,0)
        #             })
        mask_change_win = morphology(mask_change_win, area_maximum, size_kernel = 6)
        return mask_change_win, meta
    # except ValueError as error:
    #     print(error)
    

def morphology(mask_change_win, area_maximum, size_kernel):
    mask_change_win = mask_change_win.squeeze()
    contours, _ = cv2.findContours(mask_change_win.squeeze(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in tqdm(contours, desc='Run contour'):
        area = cv2.contourArea(contour)
        if area <= area_maximum:
            cv2.fillPoly(mask_change_win, [contour], 0)
            
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size_kernel,size_kernel))
    mask_small_loss = cv2.erode(mask_change_win, kernel, iterations = 1)
    mask_dialate = cv2.dilate(mask_small_loss, kernel, iterations = 1)
    
    mask_change_win = cv2.bitwise_and(mask_change_win, mask_dialate)
    return np.array([mask_change_win])

    

DICT_COLORMAP = { 
                1: (255,0,0,0),
                2: (0,255,0,0),
                3: (100, 149, 237, 0)
                }  
def export_raster(mask_change_win, meta, fp_change_out):
    with rasterio.open(fp_change_out, 'w', **meta) as dst:
        dst.write(mask_change_win)
        dst.write_colormap(1, DICT_COLORMAP)
        

def main(fp_change_out, fp_img_truoc, fp_img_sau, area_maximum):
    print(1)
    value_class_truoc = [1]
    value_class_sau = [3]
    mask1, meta = main_change_green(fp_img_truoc, fp_img_sau, value_class_truoc=value_class_truoc, value_class_sau=value_class_sau, value_change=1, area_maximum=area_maximum)
    
    print(2)
    value_class_truoc = [3]
    value_class_sau = [1]
    mask2, _ = main_change_green(fp_img_truoc, fp_img_sau, value_class_truoc=value_class_truoc, value_class_sau=value_class_sau, value_change=2, area_maximum=area_maximum)
    
    print(3)
    value_class_truoc = [3]
    value_class_sau = [2]
    mask3, _ = main_change_green(fp_img_truoc, fp_img_sau, value_class_truoc=value_class_truoc, value_class_sau=value_class_sau, value_change=3, area_maximum=area_maximum)
    
    mask_all = mask1 + mask2 + mask3
    export_raster(mask_all, meta, fp_change_out)
    print('Done')
    
    
    
if __name__=="__main__":
    area_rm = 100
    fp_change_out = f'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/img_ori/clip/RS_TEST_XOA_3857/change_with_5vs6_6_{area_rm}.tif'
    fp_img_truoc = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/2023-05_8bit_perimage/2023-05/mosaic_rs/oke/2023-05_mosaic.tif'
    fp_img_sau = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/img_ori/clip/RS_TEST_XOA_3857/2023-06_mosaic_union_ok_fix.tif'
    main(fp_change_out, fp_img_truoc, fp_img_sau, area_rm)
