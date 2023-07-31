import os, glob
import shutil

dir_img = r"/home/skm/SKM/WORK/Video_hold/img_tif"
dir_mask = r"/home/skm/SKM/WORK/Video_hold/img_tif_mask"
dir_data_train_ok = r"/home/skm/SKM/WORK/Video_hold/datatrain_and_model"

dir_data_train_img = os.path.join(dir_data_train_ok, 'img')
dir_data_train_mask = os.path.join(dir_data_train_ok, 'img_mask')
if not os.path.exists(dir_data_train_img):
    os.makedirs(dir_data_train_img)
if not os.path.exists(dir_data_train_mask):
    os.makedirs(dir_data_train_mask)

list_fp_mask = glob.glob(os.path.join(dir_mask, '*.tif'))

for fp_mask in list_fp_mask:
    name_file = os.path.basename(fp_mask)
    fp_img = os.path.join(dir_img, name_file)
    if os.path.exists(fp_img):
        shutil.copy2(fp_mask, dir_data_train_mask)
        shutil.copy2(fp_img, dir_data_train_img)

