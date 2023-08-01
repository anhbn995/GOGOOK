import os, glob
import rasterio
import numpy as np
from tqdm import tqdm

size = 64
dir_dataset = r"/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/gen_Truot_Lo_Uint8_4band_cut_64_stride_40_time_20230515_085503"
out_dir_npy = f"/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/NPY_FOR_U2net_3net/size_{size}_v2"
name_folder_tiff = ["images", "masks"]
name_folder_npy = [f"npy_i", f"npy_m"]



for folder_name in tqdm(name_folder_tiff):
    dir_img = os.path.join(dir_dataset, folder_name)
    list_tiff = glob.glob(os.path.join(dir_img, '*.tif'))
    
    dir_out = os.path.join(out_dir_npy, f"{name_folder_npy[name_folder_tiff.index(folder_name)]}")
    os.makedirs(dir_out, exist_ok=True)
    
    for fp_img in tqdm(list_tiff):
        fp_out_npy = os.path.join(dir_out, os.path.basename(fp_img).replace('.tif','.npy'))
        with rasterio.open(fp_img) as src:
            img = src.read()
        np.save(fp_out_npy, img)
        
    
