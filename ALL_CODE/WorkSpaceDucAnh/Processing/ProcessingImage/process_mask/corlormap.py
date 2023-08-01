
import rasterio
import os, glob
import numpy as np
def get_list_name_file(path_folder, name_file = '*.tif'):
    list_img_dir = []
    for file_ in glob.glob(os.path.join(path_folder, name_file)):
        _, tail = os.path.split(file_)
        list_img_dir.append(tail)
    return list_img_dir
    

def run_add_colormap(in_img, out_img):
    with rasterio.Env():
        with rasterio.open(in_img) as src:
            shade = src.read(1).astype('uint8')
            meta = src.meta.copy()
            meta.update({'nodata': 0, 'dtype':'uint8'})
        with rasterio.open(out_img, 'w', **meta) as dst:
            dst.write(shade, indexes=1)
            dst.write_colormap(
                1, {
                    0: (0, 0, 0, 0),
                    1: (34,139,34,255),
                    2: (100, 149, 237, 255), 
                    # 3: (100, 149, 237, 255), 
                    4: (101,67,33, 255)})

in_dir = r"C:\Users\SkyMap\Desktop\AAAAA"
out_dir = r"C:\Users\SkyMap\Desktop\AAAAA\A"

list_name = get_list_name_file(in_dir)
for name in list_name:
    in_path = os.path.join(in_dir, name)
    out_path = os.path.join(out_dir, name)
    run_add_colormap(in_path, out_path)
    print(1)