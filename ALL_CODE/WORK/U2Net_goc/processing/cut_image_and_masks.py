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

def ckeck_size_npy(fp_npy):
    npy = np.load(fp_npy)
    if npy.shape[0] != npy.shape[1]:
        return True
    return False

    # if (npy1.shape[0] != npy2.shape[0]) or (npy1.shape[1] != npy2.shape[1]) or (npy1.shape[2] != npy2.shape[2]):
    #     print('meos')

def remove_npy_different_shape(dir_img, dir_label):
    list_img = glob.glob(os.path.join(dir_img, '*.npy'))
    sum_rm = 0
    for fp_img in tqdm(list_img, desc='Check difference size of train or val'):
        fp_label = os.path.join(dir_label, os.path.basename(fp_img))
        if ckeck_size_npy(fp_img):
            print(os.path.basename(fp_img))
            os.remove(fp_img)
            os.remove(fp_label)
            sum_rm += 1
    print(f"Da xoa {sum_rm} file")





        
# path_img = '/media/skymap/data/tmp/Cropx2_512/images_DSx2_cut_img/*_mask.tif'
# out_path = "/media/skymap/data/tmp/Cropx2_512/ZZ/"


path_img = '/home/skm/SKM16/Work/OpenLand/3_dichHistogram/Training_Water/Data_Train_and_Model/images/*_mask.tif'
out_path = '/home/skm/SKM16/Work/OpenLand/3_dichHistogram/Training_Water/Data_Train_and_Model/npy_water_cut_512_v2/'
# size_cut = 512
# stride_ = 256

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
    
output_box = '{}_box_{}'



for image in glob.glob(path_img)[0:]:
    n=0
    print(image)
    mask_name = os.path.basename(image)
    out_name = mask_name.replace('_mask.tif', '')
    with rasterio.open(image) as ras:
        with rasterio.open(image.replace('_mask', '')) as inds:
            tile_width, tile_height = 512, 512
            # if any(ext in image for ext in ["gmap","bing","grid"]):
            if any(ext in image for ext in ["gmap","grid"]):
                stride = 512
            else:
                stride = 256
            height = inds.height
            width = inds.width

            for window, transform in get_tiles(inds, tile_width, tile_height, stride):
                if np.random.random_sample()>0.2499:
                    outpath_label = os.path.join(train_label, output_box.format(out_name,'{0:0004d}'.format(n)))
                    outpath_image = os.path.join(train_image, output_box.format(out_name,'{0:0004d}'.format(n)))
                else:
                    outpath_label = os.path.join(val_label, output_box.format(out_name,'{0:0004d}'.format(n)))
                    outpath_image = os.path.join(val_image, output_box.format(out_name,'{0:0004d}'.format(n)))
                img = inds.read(window=window)
                lab = ras.read(window=window)
                if np.count_nonzero(lab):# or np.random.uniform()< 0.01:
                    lab[lab!=255]=0
                    np.save(outpath_label, lab.transpose(1,2,0))
                    np.save(outpath_image, img.transpose(1,2,0))
                    n+=1
        print(n)

# count rs
print(train_image, len(glob.glob(os.path.join(train_image, '*.npy'))))
print(train_label, len(glob.glob(os.path.join(train_label, '*.npy'))))
print(val_image, len(glob.glob(os.path.join(val_image, '*.npy'))))
print(val_label, len(glob.glob(os.path.join(val_label, '*.npy'))))

# recheck rs
# remove_npy_different_shape(train_image, train_label)
# remove_npy_different_shape(val_image, val_label)
