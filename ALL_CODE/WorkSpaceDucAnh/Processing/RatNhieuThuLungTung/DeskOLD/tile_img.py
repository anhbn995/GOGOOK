import affine
from pydantic import main
import rasterio
import os, glob
from tqdm import tqdm
import numpy as np
from rio_tiler.io import COGReader

def get_list_name_file(path_folder, name_file = '*.tif'):
    list_img_dir = []
    for file_ in glob.glob(os.path.join(path_folder, name_file)):
        _, tail = os.path.split(file_)
        list_img_dir.append(tail)
    return list_img_dir

def write_image(data, height, width, numband, crs, tr, out, type=None):
    """
        Export numpy array to image by rasterio.
    """
    if type:
        dtype_custom = type
        data = data.astype(dtype_custom)
    else:
        dtype_custom = data.dtype
    with rasterio.open(
                        out,
                        'w',
                        driver='GTiff',
                        height=height,
                        width=width,
                        count=numband,
                        dtype=dtype_custom,
                        crs=crs,
                        transform=tr,
                        # nodata=0,
                        ) as dst:
                            dst.write(data)

def upsample_by_tile_image(in_img, out_resample, size = 3):
    src = rasterio.open(in_img)
    crs = src.crs
    tr = src.transform
    tr_new = affine.Affine(tr[0]*size, tr[1], tr[2], tr[3], tr[4]*size, tr[5])

    bbox = tuple(src.bounds)
    h, w = src.height, src.width
    max_size = int(max(h/size, w/size))
    index = list(range(1, src.count + 1))
    with COGReader(in_img) as image:
        # img, mask = image.part(bbox=bbox, bounds_crs=crs, max_size= max_size)
        img, mask = image.part(bbox=bbox, bounds_crs=crs, max_size= max_size, indexes=index)
    write_image(img, h/size, w/size, src.count, crs, tr_new, out_resample, type=None)



in_dir = r"C:\Users\SkyMap\Desktop\PixelMagic"
out_dir = r"C:\Users\SkyMap\Desktop\PixelMagic\rs"
size = 3

list_name = get_list_name_file(in_dir)
for name in tqdm(list_name):
    in_fp = os.path.join(in_dir, name)
    out_fp = os.path.join(out_dir, name)
    upsample_by_tile_image(in_fp, out_fp, size)
