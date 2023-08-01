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
        # print(window)
        transform = windows.transform(window, ds.transform)
        yield window, transform

def get_tiles_ngt(ds, width, height):
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
    if len(offset) > 50:
        offset = random.sample(offset, 50)

    offset = set(offset)
    for col_off, row_off in offset: 
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform


if __name__ == '__main__':   
    
    # out_path_image = '/media/skymap/Learnning/public/USA/Queensland Mosaics/road_polygon/data_1/image'
    # out_path_label = '/media/skymap/Learnning/public/USA/Queensland Mosaics/road_polygon/data_1/shape'

    # img_path = '/media/skymap/Learnning/public/USA/Queensland Mosaics/road_polygon/img/*mask.tif'

    # out_path_image = r'E:\AAA\BaiBoiVenSong\Backup_DNN\1img\img_convert_01_float\img_01_mask_0_to_11_train\Train\npy_01_Balat\img'
    # out_path_label = r'E:\AAA\BaiBoiVenSong\Backup_DNN\1img\img_convert_01_float\img_01_mask_0_to_11_train\Train\npy_01_Balat\mask'
    # import os
    # if not os.path.exists(out_path_image):
    #     os.makedirs(out_path_image) 
    # if not os.path.exists(out_path_label):
    #     os.makedirs(out_path_label) 

    # img_path = r'E:\AAA\BaiBoiVenSong\Backup_DNN\1img\img_convert_01_float\img_01_mask_0_to_11_train\Train\BaLat\*mask.tif'


    # out_path_image = r'E:\AAA\BaiBoiVenSong\Data_train_oke\Uint8\data_train\img_64'
    # out_path_label = r'E:\AAA\BaiBoiVenSong\Data_train_oke\Uint8\data_train\mask_64'

    # img_path = r'E:\AAA\BaiBoiVenSong\Data_train_oke\Uint8\img_and_mask\*mask.tif'
    # out_path_image = r'E:\AAA\BaiBoiVenSong\Data_train_oke\Uint8\img_and_mask\tmp\img_64'
    # out_path_label = r'E:\AAA\BaiBoiVenSong\Data_train_oke\Uint8\img_and_mask\tmp\mask_64'

    # img_path = r'/home/skm/SKM_OLD/ZZ_ZZ/DuongBoVenBien/img_origin/*mask.tif'
    # out_path_image = r'/home/skm/SKM_OLD/ZZ_ZZ/DuongBoVenBien/img_256'
    # out_path_label = r'/home/skm/SKM_OLD/ZZ_ZZ/DuongBoVenBien/mask_256'
    """Bo da"""
    import time
    # x = time.time()
    # img_path = r'/home/skm/SKM_OLD/ZZ_ZZ/cloud_shadow/img_origin/*mask.tif'
    # out_path_image = r'/home/skm/SKM_OLD/ZZ_ZZ/model_segmentation/V2/Data_Train_V2/256/npy_i2_256'
    # out_path_label = r'/home/skm/SKM_OLD/ZZ_ZZ/model_segmentation/V2/Data_Train_V2/256/npy_m2_256'
    # size_crop = 256

    import time
    x = time.time()
    img_path = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Gen_for_u2net/Img_and_mask/*mask.tif'
    name_folder = "gen_numpy_cut128_uint8"
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
    output_box = 'box_{}'
    print(glob.glob(img_path))
    for img in glob.glob(img_path):
        print(img)
        with rasterio.open(img.split('.')[0][:-5]+'.tif') as ras:
            with rasterio.open(img) as inds:
                tile_width, tile_height = size_crop, size_crop
                for window, transform in get_tiles(inds, tile_width, tile_height):

                    outpath_label = os.path.join(out_path_label, output_box.format('{0:003d}'.format(n))+'.npy')
                    outpath_image = os.path.join(out_path_image, output_box.format('{0:003d}'.format(n))+'.npy')

                    if np.count_nonzero(inds.read(window=window)) > 10 :#or (np.random.uniform()< 0.5):
                        with open(outpath_image, 'wb') as f:
                            np.save(f, ras.read(window=window))

                        with open(outpath_label, 'wb') as l:
                            np.save(l, inds.read(window=window))
                        # print(inds.read(window=window).shape)
                        n+=1


#  sinh them negative
    # k = len(glob.glob(out_path_label + '/*.npy')) + 2
    # print(k)
    # output_box = 'box_{}'
    # print(glob.glob(img_path)[-1])
    # for img in glob.glob(img_path):
    #     print(img)
    #     with rasterio.open(img.split('.')[0][:-5]+'.tif') as ras:
    #         with rasterio.open(img) as inds:
    #             tile_width, tile_height = 256, 256
    #             for window, transform in get_tiles_ngt(inds, tile_width, tile_height):

    #                 outpath_label = os.path.join(out_path_label, output_box.format('{0:003d}'.format(k))+'.npy')
    #                 outpath_image = os.path.join(out_path_image, output_box.format('{0:003d}'.format(k))+'.npy')

    #                 if np.count_nonzero(inds.read(window=window)) == 0:
    #                     with open(outpath_image, 'wb') as f:
    #                         np.save(f, ras.read(window=window))

    #                     with open(outpath_label, 'wb') as l:
    #                         np.save(l, inds.read(window=window))
    #                     k+=1
    print(time.time()-x)