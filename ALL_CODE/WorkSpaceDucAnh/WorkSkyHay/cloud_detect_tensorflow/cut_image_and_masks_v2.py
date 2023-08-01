import rasterio.mask
import rasterio
from rasterio import windows
from itertools import product
import numpy as np
import glob, os
from tqdm import tqdm
import sys
import random

"""
path_img: folder containing the input data
*Note:  + in folder have both image original xxx.tif and image mask xxx_mask.tif
        + path image passed in is the format  $PATH_IMAGE/*_mask.tif

out_path: folder containing the output data
"""

def get_tiles(ds, width, height, stride):
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    offset = []
    for col_off, row_off in offsets:
        if row_off + width > nrows:
            row_off = nrows - width
        if  col_off + height > nols:
            col_off = nols - height
        offset.append((col_off, row_off))
    offset = set(offset)
    for col_off, row_off in tqdm(offset): 
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform
        
        
# path_img = r'/home/geoai/eodata/cloud_detect_tensorflow/raw_final/shadow_data/all/*_label.tif'
# out_path = r'/home/geoai/eodata/cloud_detect_tensorflow/shadow_only/data2/'

# path_img = r"/home/geoai/eodata/cloud_detect_tensorflow/raw_final/shadow_data/all/contain/*_label.tif"
# out_path = r"/home/geoai/eodata/cloud_detect_tensorflow/shadow_only/data_only_nodata/"

# path_img = r"/home/geoai/eodata/cloud_detect_tensorflow/raw_img/shadow/*_label.tif"
# out_path = r'/home/geoai/eodata/cloud_detect_tensorflow/cloud_and_shadow/data/'

# path_img = r"/home/geoai/eodata/cloud_detect_tensorflow/raw_final/noise_data/img/add/*_label.tif"
# out_path = r"/home/geoai/eodata/cloud_detect_tensorflow/cloud_noise/data/"

path_img = r"/home/geoai/eodata/cloud_detect_tensorflow/raw_final/cloud_data/all/mau2/*_label.tif"
out_path = r"/home/geoai/eodata/cloud_detect_tensorflow/cloud_only_v2/data/"

n=300

def mk_dir(path_image, name1, name2):
    if not os.path.exists(path_image+name1):
        os.mkdir(path_image+name1)
    if not os.path.exists(path_image+name2):
        os.mkdir(path_image+name2)
    return path_image+name1, path_image+name2

if not os.path.exists(out_path):
    os.mkdir(out_path)
path_train, path_val = mk_dir(out_path, 'train/', 'val/')
train_image, train_label = mk_dir(path_train, 'image/', 'label/')
val_image, val_label = mk_dir(path_val, 'image/', 'label/')

output_box = 'box_{}'

for image in glob.glob(path_img):
    with rasterio.open(image) as ras:
        with rasterio.open(image.replace('_label', '')) as inds:
            tile_width, tile_height = 512, 512 #512, 512
            stride = tile_width//2
            height = inds.height
            width = inds.width

            for window, transform in get_tiles(inds, tile_width, tile_height, stride):
                if np.random.random_sample() > 0.09999:
                    outpath_label = os.path.join(train_label, output_box.format('{0:0004d}'.format(n))+".tif")
                    outpath_image = os.path.join(train_image, output_box.format('{0:0004d}'.format(n))+".tif")
                else:
                    outpath_label = os.path.join(val_label, output_box.format('{0:0004d}'.format(n))+".tif")
                    outpath_image = os.path.join(val_image, output_box.format('{0:0004d}'.format(n))+".tif")
                img = inds.read(window=window)
                # if (img==0).sum() == 0: continue
                if img.sum() == 0:
                    # if np.random.randint(0,100)<5: pass
                    # else: continue
                    continue
                lab = ras.read(window=window)
                
                #write image
                out_meta = inds.meta
                out_meta.update({"driver": "GTiff",
                        "height": img.shape[1],
                        "width": img.shape[2],
                        "transform": transform})
                with rasterio.open(outpath_image, "w", compress='lzw', **out_meta) as dest:
                    dest.write(img)

                # write mask label
                out_meta1 = ras.meta
                out_meta1.update({"driver": "GTiff",
                        "height": img.shape[1],
                        "width": img.shape[2],
                        "transform": transform,
                        "nodata" : None})
                with rasterio.open(outpath_label, "w", compress='lzw', **out_meta1) as dest:
                    dest.write(lab.astype(np.uint8))
                n=n+1
        print("Finished create {} image!!".format(n))