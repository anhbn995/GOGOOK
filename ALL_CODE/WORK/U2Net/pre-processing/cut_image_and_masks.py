import rasterio.mask
import rasterio
from rasterio import windows
from itertools import product
import numpy as np
import glob, os
from tqdm import tqdm

"""
path_img: folder containing the input data
*Note:  + in folder have both image original xxx.tif and image mask xxx_mask.tif
        + path image passed in is the format  $PATH_IMAGE/*_mask.tif

out_path: folder containing the output data
├── train
│   ├── image
│   │   ├── file .npy
│   ├── label
│   │   ├── file .npy
├── val
│   ├── image
│   │   ├── file .npy
│   ├── label
│   │   ├── file .npy
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

path_img = r'/home/skm/SKM16/Work/SonalPanel_ThaiLand/1Ver2_lable2/images_8bit_perimage/Data_Train_and_Model/boTrainU2net/*_mask.tif'
out_path = r"/home/skm/SKM16/Work/SonalPanel_ThaiLand/1Ver2_lable2/images_8bit_perimage/Data_Train_and_Model/U2net_Ds/"


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
    print(image)
    with rasterio.open(image) as ras:
        with rasterio.open(image.replace('_mask', '')) as inds:
            tile_width, tile_height = 256, 256
            stride = 200
            height = inds.height
            width = inds.width

            for window, transform in get_tiles(inds, tile_width, tile_height, stride):
                if np.random.random_sample()>0.2499:
                    outpath_label = os.path.join(train_label, output_box.format('{0:0004d}'.format(n)))
                    outpath_image = os.path.join(train_image, output_box.format('{0:0004d}'.format(n)))
                else:
                    outpath_label = os.path.join(val_label, output_box.format('{0:0004d}'.format(n)))
                    outpath_image = os.path.join(val_image, output_box.format('{0:0004d}'.format(n)))
                img = inds.read(window=window)
                lab = ras.read(window=window)
                if np.count_nonzero(lab):# or np.random.uniform()< 0.01:
                    lab[lab!=1]=0
                    np.save(outpath_label, lab.transpose(1,2,0))
                    np.save(outpath_image, img.transpose(1,2,0))
                    n+=1
        print(n)