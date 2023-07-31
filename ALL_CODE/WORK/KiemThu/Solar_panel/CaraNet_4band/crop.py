import os
import rasterio
from glob import glob
from tqdm import tqdm
from rasterio.windows import Window


def get_window_for_img(name_file, h_end, w_end, crop_size, stride_size):
    size_win = crop_size - stride_size
    if ("_w0" in name_file) and ("_h0" in name_file):
        win = Window(0, 0, size_win+stride_size/2, size_win+stride_size/2)
        return win
    elif (f"_w{w_end}" in name_file) and ("_h0" in name_file):
        win = Window(stride_size/2, 0, size_win+stride_size/2, size_win+stride_size/2)
        return win
    elif (f"_w0" in name_file) and ("_h{h_end}" in name_file):
        win = Window(0, stride_size/2, size_win+stride_size/2, size_win+stride_size/2)
        return win
    elif (f"_w{w_end}" in name_file) and ("_h{h_end}" in name_file):
        win = Window(stride_size/2, stride_size/2, size_win+stride_size/2, size_win+stride_size/2)
        return win

    elif ("_w0" in name_file) and ("_h0" not in name_file) and ("_h{h_end}" not in name_file):
        win = Window(0, stride_size/2, size_win + stride_size/2, size_win)
        return win
    elif (f"_w{w_end}" in name_file) and ("_h0" not in name_file) and ("_h{h_end}" not in name_file):
        win = Window(stride_size/2, stride_size/2, size_win + stride_size/2, size_win)
        return win
    elif ("_h0" in name_file) and ("_w0" not in name_file) and (f"_w{w_end}" not in name_file):
        win = Window(stride_size/2, 0, size_win, size_win + stride_size/2)
        return win
    elif ("_h{h_end}" in name_file) and (f"_w0" not in name_file) and (f"_w{w_end}" not in name_file):
        win = Window(stride_size/2, stride_size/2, size_win, size_win + stride_size/2)
        return win
    else:
        win = Window(stride_size/2, stride_size/2, size_win, size_win)
        return win


def crop_mask_for_mosaic(image_path, outdir_crop, crop_size, stride_size):
    name_base = os.path.basename(image_path)

    with rasterio.open(image_path) as src:
        h,w = src.height,src.width
        list_weight = list(range(0, w, stride_size))
        list_hight = list(range(0, h, stride_size))
        meta = src.meta
        win = get_window_for_img(name_base, list_hight[-1], list_weight[-1], crop_size, stride_size)
        # print(win, "aaaaaaaaaaaa")
        img_window_crop  = src.read(window=win)
        win_transform = src.window_transform(win)
        meta.update({'height': win.height, 'width': win.width, 'transform':win_transform, 'nodata': 0, 'count':1})
        fp_out = os.path.join(outdir_crop, name_base)
        with rasterio.open(fp_out, 'w',**meta) as dst:
            dst.write(img_window_crop, window=Window(0, 0, img_window_crop.shape[2], img_window_crop.shape[1]))


def main_crop_mask(list_name, dir_predict):
    # dir_predict = r"/home/skm/SKM16/Work/SonalPanel_ThaiLand/1Ver2_lable2/izmages_8bit_perizmage/images_per95/tmp_forpredict_big/CaraNet-best-solar-882022Sep06-14h57m48s"
    # list_name = [   
    #     "01_July_Mosaic_P_2",
    #     "01_July_Mosaic_P_3",
    #     "01_July_Mosaic_P_4",
    #     "01_July_Mosaic_P_5",
    #     "01_July_Mosaic_P_6",
    #     "02_May_Mosaic_P_2",
    #     "03_July_Mosaic_P_2"
    #     ]

    outdir_crop = dir_predict + "crop_predict"
    os.makedirs(outdir_crop, exist_ok=True)
    crop_size = 512
    stride_size = 256
    for sub_str in tqdm(list_name):
        list_fp = glob(os.path.join(dir_predict, "*.tif"))
        list_ones_img = [s for s in list_fp if sub_str in s]
        outdir_crop_img = os.path.join(outdir_crop, sub_str)
        os.makedirs(outdir_crop_img, exist_ok=True)
        for fp_mask in tqdm(list_ones_img):
            crop_mask_for_mosaic(fp_mask, outdir_crop_img, crop_size, stride_size)
    return outdir_crop


