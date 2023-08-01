import rasterio
import numpy as np
from rasterio.windows import Window
from tqdm import tqdm

def write_window_many_chanel(output_ds, arr_c, window_):
    """
        Write imge multi band with window size
    """
    for c, arr in enumerate(arr_c):
        output_ds.write(arr, window = window_, indexes= c + 1)

def remove_nan_value(image_path, new_image, win_size=500):
    with rasterio.open(image_path) as src:
        h,w = src.height, src.width
        profile = src.meta.copy()
        numband = src.count

    list_w_off = list(range(0, w, win_size))
    list_h_off = list(range(0, h, win_size))

    with rasterio.open(new_image, 'w', **profile) as dst:
        outdstput_ds = np.empty((numband,h,w))
        
    src = rasterio.open(image_path)
    with rasterio.open(new_image, 'r+') as dst:
        with tqdm(total=len(list_w_off)*len(list_h_off)) as pbar:
            for w_off in list_w_off:
                for h_off in list_h_off:
                    if w - w_off < win_size and h - h_off < win_size:
                        size_w = w - w_off
                        size_h = h - h_off
                        img_window = src.read(window=Window(w_off, h_off, size_w, size_h))
                        img_window[np.isnan(img_window)] = 0
                        write_window_many_chanel(dst, img_window, Window(w_off, h_off, size_w, size_h))
                    elif w - w_off < win_size:
                        size_w = w - w_off
                        size_h = win_size
                        img_window = src.read(window=Window(w_off, h_off, size_w, size_h))
                        img_window[np.isnan(img_window)] = 0
                        write_window_many_chanel(dst, img_window, Window(w_off, h_off, size_w, size_h))
                    elif h - h_off < win_size:
                        size_h = h - h_off
                        size_w = win_size
                        img_window = src.read(window=Window(w_off, h_off, size_w, size_h))
                        img_window[np.isnan(img_window)] = 0
                        write_window_many_chanel(dst, img_window, Window(w_off, h_off, size_w, size_h))
                    else:
                        size_w = win_size
                        size_h = win_size
                        img_window = src.read(window=Window(w_off, h_off, size_w, size_h))
                        img_window[np.isnan(img_window)] = 0
                        write_window_many_chanel(dst, img_window, Window(w_off, h_off, size_w, size_h))
                    pbar.update()

if __name__ == "__main__":
    image_path=r"C:\Users\SkyMap\Desktop\afc15c2be0534599a3105cccc475c2d8.tif"
    new_image=r"E:\farm\zzzzzza.tif"
    remove_nan_value(image_path, new_image)
