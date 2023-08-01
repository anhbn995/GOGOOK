import rasterio.mask
import rasterio
from rasterio import windows
from itertools import product
import numpy as np
import glob, os
from tqdm import tqdm
import sys

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
        
        
# path_img = '/home/skm/SKM/WORK/Demo_Kuwait/Data_Train_and_Model/openland_dstraining_v3_u2net/Img_cut_img/*_mask.tif'
# out_path = '/home/skm/SKM/WORK/Demo_Kuwait/Data_Train_and_Model/openland_dstraining_v3_u2net/dataset_training_u2net/'

# path_img = r'D:\scanmap_test_crop\tmp\image_cut_img\*_mask.tif'
# out_path = r"D:\scanmap_test_crop\tmp\data_final"

path_img = r'/home/geoai/eodata/cloud_detect_tensorflow/raw_more/shadow/*_label.tif'
out_path = r'/home/geoai/eodata/cloud_detect_tensorflow/cloud_and_shadow/data/'

# path_img = r"/home/geoai/eodata/cloud_detect_tensorflow/raw_img/shadow/*_label.tif"
# out_path = r'/home/geoai/eodata/cloud_detect_tensorflow/cloud_and_shadow/data/'

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

n=0
for image in glob.glob(path_img):
    with rasterio.open(image) as ras:
        with rasterio.open(image.replace('_label', '')) as inds:
            tile_width, tile_height = 512, 512
            stride = tile_width//2
            height = inds.height
            width = inds.width

            for window, transform in get_tiles(inds, tile_width, tile_height, stride):
                if np.random.random_sample()>0.0999:
                    outpath_label = os.path.join(train_label, output_box.format('{0:0004d}'.format(n))+".tif")
                    outpath_image = os.path.join(train_image, output_box.format('{0:0004d}'.format(n))+".tif")
                else:
                    outpath_label = os.path.join(val_label, output_box.format('{0:0004d}'.format(n))+".tif")
                    outpath_image = os.path.join(val_image, output_box.format('{0:0004d}'.format(n))+".tif")
                img = inds.read(window=window)
                lab = ras.read(window=window)
                #write image
                out_meta = inds.meta
                out_meta.update({"driver": "GTiff",
                        "height": img.shape[1],
                        "width": img.shape[2],
                        "transform": transform})
                with rasterio.open(outpath_image, "w", compress='lzw', **out_meta) as dest:
                    dest.write(img)

                #write mask label
                out_meta1 = ras.meta
                out_meta1.update({"driver": "GTiff",
                        "height": img.shape[1],
                        "width": img.shape[2],
                        "transform": transform,
                        "nodata" : None})##################
                # print(ras.meta)
                # sys.exit(0)
                with rasterio.open(outpath_label, "w", compress='lzw', **out_meta1) as dest: # count =??
                    dest.write(lab.astype(np.uint8))######################
                n=n+1
        print(n)