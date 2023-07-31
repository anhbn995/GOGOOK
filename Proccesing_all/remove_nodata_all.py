import rasterio
import numpy as np
import os, glob

def get_list_fp(folder_dir, type_file = '*.tif'):
        """
            Get all file path with file type is type_file.
        """
        list_fp = []
        for file_ in glob.glob(os.path.join(folder_dir, type_file)):
            head, tail = os.path.split(file_)
            # list_fp.append(os.path.join(head, tail))
            list_fp.append(tail)
        return list_fp

# dir_img = r'/home/skm/SKM_OLD/WORK/MongCo/img_v2_crop'
# dir_mask = r'/home/skm/SKM_OLD/WORK/MongCo/img_v2_mask_crop'

dir_img = r'/home/skm/SKM16/Work/Npark_planet2/Data_train_uint8_water/Data_train_cut_512/img_cut_img_crop'
dir_mask = r'/home/skm/SKM16/Work/Npark_planet2/Data_train_uint8_water/Data_train_cut_512/img_cut_img_mask_crop'
list_fn = get_list_fp(dir_img)
# i = 0
for fn in list_fn:
# fn = 'img_107.tif'
    fp_img = os.path.join(dir_img, fn)
    fp_mask = os.path.join(dir_mask, fn)
    with rasterio.open(fp_mask) as src:
        band = src.read()
        # print(band)
    if np.all(band == 0):
        # i += 1
        # print(i+1)
        os.remove(fp_img)
        os.remove(fp_mask)
print("File Removed!")
# else:
#     continue
# source activate mlenv
# bash local_defores_generator.sh