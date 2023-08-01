import rasterio
import glob, os
import numpy as np
from tqdm import tqdm

def mask1_tru_mask2(fp_out, fp_mask1, fp_mask2, value_mask1=255, value_mask2=255, value_mask_return = 255):
    with rasterio.open(fp_mask1) as src1:
        mask1 = src1.read()
        meta = src1.meta
    with rasterio.open(fp_mask2) as src2:
        mask2 = src2.read()

    # lấy giá trị của lớp
    index_mask1 = np.where(mask1 == value_mask1)
    index_mask2 = np.where(mask2 == value_mask2)

    # tao ket quả
    mask_rs = np.zeros_like(mask1)
    mask_rs[index_mask1] = value_mask_return
    mask_rs[index_mask2] = 0

    with rasterio.open(fp_out, 'w', **meta) as dst:
        dst.write(mask_rs)

if __name__=='__main__':
    dir_mask1 = r'D:\Tmp_DrawLabel\Maluku_Deforestation\2022_rs'
    dir_mask2 = r'D:\Tmp_DrawLabel\Maluku_Deforestation\2022_shp_mask'
    out_dir = r"D:\Tmp_DrawLabel\Maluku_Deforestation\2022_da_xoa_bot"
    
    os.makedirs(out_dir, exist_ok=True)
    for fp in tqdm(glob.glob(os.path.join(dir_mask1, '*.tif'))):
        fname = os.path.basename(fp)
        fp_mask1 = os.path.join(dir_mask1, fname)
        fp_mask2 = os.path.join(dir_mask2, fname)
        fp_out = os.path.join(out_dir, fname)
        mask1_tru_mask2(fp_out, fp_mask1, fp_mask2, value_mask1=0, value_mask2=255, value_mask_return = 255)
        print(f'Done {fname}')






