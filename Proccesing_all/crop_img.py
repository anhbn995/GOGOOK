import rasterio.mask
import rasterio
from rasterio import windows
from itertools import product
import numpy as np
import glob, os
import sys
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
import fiona
n=0


def get_tiles(ds, width, height):
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
    offset = set(offset)
    for col_off, row_off in offset: 
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        # print(window)
        transform = windows.transform(window, ds.transform)
        yield window, transform
if __name__ == '__main__':    
    # out_path_image = r'E:\AAA\\Data_train_oke\Uint8\img_and_mask\tmp\img_64'
    # out_path_label = r'E:\AAA\\Data_train_oke\Uint8\img_and_mask\tmp\mask_64'

    out_path_image = r"/home/skm/SKM16/Work/SonalPanel_ThaiLand/1Ver2_lable2/images_8bit_perimage/Data_Train_and_Model/U2net_Ds/img_512"
    out_path_label = r"/home/skm/SKM16/Work/SonalPanel_ThaiLand/1Ver2_lable2/images_8bit_perimage/Data_Train_and_Model/U2net_Ds/mask_512"
    import os
    if not os.path.exists(out_path_image):
        os.makedirs(out_path_image) 
    if not os.path.exists(out_path_label):
        os.makedirs(out_path_label) 

    img_path = r'/home/skm/SKM16/Work/SonalPanel_ThaiLand/1Ver2_lable2/images_8bit_perimage/Data_Train_and_Model/boTrainU2net/*mask.tif'


    output_box = 'box_{}'
    print(glob.glob(img_path))
    for img in glob.glob(img_path):
        with rasterio.open(img.split('.')[0][:-5]+'.tif') as ras:
            with rasterio.open(img) as inds:
                tile_width, tile_height = 512, 512
                for window, transform in get_tiles(inds, tile_width, tile_height):

                    outpath_label = os.path.join(out_path_label, output_box.format('{0:003d}'.format(n))+'.npy')
                    outpath_image = os.path.join(out_path_image, output_box.format('{0:003d}'.format(n))+'.npy')

                    if np.count_nonzero(inds.read(window=window)) > 1000:
                        with open(outpath_image, 'wb') as f:
                            np.save(f, ras.read(window=window))

                        with open(outpath_label, 'wb') as l:
                            np.save(l, inds.read(window=window))
                        print(inds.read(window=window).shape)
                        n+=1