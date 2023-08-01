import rasterio.mask
import rasterio
from rasterio import windows
from itertools import product
import numpy as np
import glob, os
# import sys
# from multiprocessing.pool import Pool
# from functools import partial
# import multiprocessing
# import fiona
import random

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


def get_tiles(ds, width, height):
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, 100), range(0, nrows, 100))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    offset = []
    for col_off, row_off in offsets:
        if row_off + tile_width > nrows:
            row_off = nrows - tile_width
        if  col_off + tile_height > nols:
            col_off = nols - tile_height
        offset.append((col_off, row_off))
    offset = set(offset)
    for col_off, row_off in offset: 
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform


def get_tiles_ngt(ds, width, height):
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, 200), range(0, nrows, 200))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    offset = []
    for col_off, row_off in offsets:
        if row_off + tile_width > nrows:
            row_off = nrows - tile_width
        if  col_off + tile_height > nols:
            col_off = nols - tile_height
        offset.append((col_off, row_off))
    if len(offset) > 300:
        offset = random.sample(offset, 300)

    offset = set(offset)
    for col_off, row_off in offset: 
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform


if __name__ == '__main__':   
    import time
    x = time.time()
    # img_path = r'/home/skm/SKM_OLD/ZZ_ZZ/cloud_shadow/create_data_train_uint8/img_origin/*mask.tif'
    # out_path_image = r'/home/skm/SKM_OLD/ZZ_ZZ/cloud_shadow/create_data_train_uint8/unet_xz_256_moi/img_256'
    # out_path_label = r'/home/skm/SKM_OLD/ZZ_ZZ/cloud_shadow/create_data_train_uint8/unet_xz_256_moi/mask_256'
    # size_crop = 256

    img_path = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Gen_for_u2net/Img_and_mask/*mask.tif'
    name_folder = "v1"
    out_path_image = f'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Gen_for_u2net/{name_folder}/npy_i2_256'
    out_path_label = f'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Gen_for_u2net/{name_folder}/npy_m2_256'
    size_crop = 128

    import os
    if not os.path.exists(out_path_image):
        os.makedirs(out_path_image) 
    if not os.path.exists(out_path_label):
        os.makedirs(out_path_label) 

# chuaanr
    n = 0
    k = 0
    output_box = 'box_{}'
    # print(glob.glob(img_path))
    for img in glob.glob(img_path):
        print(img)
        with rasterio.open(img.split('.')[0][:-5]+'.tif') as ras:
            crs = ras.crs
            with rasterio.open(img) as inds:
                tile_width, tile_height = size_crop, size_crop
                for window, transform in get_tiles(inds, tile_width, tile_height):

                    outpath_label = os.path.join(out_path_label, output_box.format('{0:003d}'.format(n))+'.tif')
                    outpath_image = os.path.join(out_path_image, output_box.format('{0:003d}'.format(n))+'.tif')

                    if (np.count_nonzero(inds.read(window=window)) > 10):
                        write_image(ras.read(window=window), size_crop, size_crop, 4, crs, transform, outpath_image, type=None)
                        write_image(inds.read(window=window), size_crop, size_crop, 1, crs, transform, outpath_label, type=None)
                        n+=1
                    if (np.count_nonzero(inds.read(window=window)) == 0) and (np.random.uniform()< 0.01):
                        write_image(ras.read(window=window), size_crop, size_crop, 4, crs, transform, outpath_image, type=None)
                        write_image(inds.read(window=window), size_crop, size_crop, 1, crs, transform, outpath_label, type=None)
                        n+=1
                        # k+=1
    # print(n, k)


#  sinh them negative
    # k = len(glob.glob(out_path_label + '/*.tif')) + 2
    # print(k)
    # output_box = 'box_{}'
    # for img in glob.glob(img_path):
    #     print(img)
    #     with rasterio.open(img.split('.')[0][:-5]+'.tif') as ras:
    #         crs = ras.crs
    #         with rasterio.open(img) as inds:
    #             tile_width, tile_height = size_crop, size_crop
    #             for window, transform in get_tiles_ngt(inds, tile_width, tile_height):

    #                 outpath_label = os.path.join(out_path_label, output_box.format('{0:003d}'.format(k))+'.tif')
    #                 outpath_image = os.path.join(out_path_image, output_box.format('{0:003d}'.format(k))+'.tif')
    #                 if np.count_nonzero(inds.read(window=window)) == 0:
    #                     write_image(ras.read(window=window), size_crop, size_crop, 3, crs, transform, outpath_image, type=None)
    #                     write_image(inds.read(window=window), size_crop, size_crop, 1, crs, transform, outpath_label, type=None)
    #                     k+=1
    # print(time.time()-x)


