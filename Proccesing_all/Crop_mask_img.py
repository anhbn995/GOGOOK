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
    offsets = product(range(0, nols, 60), range(0, nrows, 60))
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
    
    size_crop = 128
    out_path_image = f'/home/skm/SKM_OLD/public/changedetection_SAR/DA/Gen_mask/data_train/img_{size_crop}'
    out_path_label = f'/home/skm/SKM_OLD/public/changedetection_SAR/DA/Gen_mask/data_train/mask_{size_crop}'

    import os
    if not os.path.exists(out_path_image):
        os.makedirs(out_path_image) 
    if not os.path.exists(out_path_label):
        os.makedirs(out_path_label) 

    img_path = r'/home/skm/SKM_OLD/public/changedetection_SAR/DA/Gen_mask/data/*mask.tif'
    

    output_box = 'box_{}'
    print(glob.glob(img_path))
    for img in glob.glob(img_path):
        with rasterio.open(img.split('.')[0][:-5]+'.tif') as ras:
            crs = ras.crs
            with rasterio.open(img) as inds:
                tile_width, tile_height = size_crop, size_crop
                for window, transform in get_tiles(inds, tile_width, tile_height):
                    outpath_label = os.path.join(out_path_label, output_box.format('{0:003d}'.format(n))+'.tif')
                    outpath_image = os.path.join(out_path_image, output_box.format('{0:003d}'.format(n))+'.tif')

                    if np.count_nonzero(inds.read(window=window)) > 10:
                        write_image(ras.read(window=window), size_crop, size_crop, 4, crs, transform, outpath_image, type=None)
                        write_image(inds.read(window=window), size_crop, size_crop, 1, crs, transform, outpath_label, type=None)

                        # with open(outpath_image, 'wb') as f:
                        #     np.save(f, ras.read(window=window))

                        # with open(outpath_label, 'wb') as l:
                        #     np.save(l, inds.read(window=window))
                        print(inds.read(window=window).shape)
                        n+=1