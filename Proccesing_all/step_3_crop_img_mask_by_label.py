import rasterio
import os, glob
from tqdm import tqdm
import numpy as np

def get_list_name_file(path_folder, name_file = '*.tif'):
    list_img_dir = []
    for file_ in glob.glob(os.path.join(path_folder, name_file)):
        _, tail = os.path.split(file_)
        list_img_dir.append(tail)
    return list_img_dir

def remove_data(img_path, mask_path):
    mask = rasterio.open(mask_path).read()
    values, counts = np.unique(mask, return_counts=True)
    print(values)
    count_255 = counts[np.where(values==255)][0]
    # if count_255 < 10:
    #     os.remove(img_path)
    #     os.remove(mask_path)

img_dir = r"/home/skm/SKM_OLD/public/changedetection_SAR/DA/Stack_Uint8_crop"
mask_dir = r"/home/skm/SKM_OLD/public/changedetection_SAR/DA/Stack_Uint8_mask_crop"

list_name = get_list_name_file(mask_dir)
i = 0
for name in tqdm(list_name):
    i = i + 1
    if i > 8000:
        break
    img_path = os.path.join(img_dir, name)
    mask_path = os.path.join(mask_dir, name)
    remove_data(img_path, mask_path)
