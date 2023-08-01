import glob, os
import rasterio
import numpy as np

def change_idx_band(fp_out, fp_img_in, list_idx_band_change):
    """Thay doi thu tu band

    Args:
        fp_out (str): duong dan anh dau ra cua anh sau khi thay doi thu tu band
        fp_img_in (str): duong dan anh dau vao, anh muon thay doi thu tu band
        list_idx_band_change (list): list indx cua thu tu band moi vd muoon chuyen anh ban dau voi band 3 len dau  va anh co 4 band thi list_idx_band_change = [3,2,1,4] (index o day da cong them 1)
    """
    os.makedirs(os.path.dirname(fp_out), exist_ok=True)
    with rasterio.open(fp_img_in) as src:
        img = src.read()
        meta = src.meta
    list_idx_band_change = (np.array(list_idx_band_change) - 1).tolist()
    img_new = img[list_idx_band_change, :]
    with rasterio.open(fp_out, 'w', **meta) as dst:
        dst.write(img_new) 
    print(f"Done: {os.path.basename(fp_img_in)}")
    
if __name__=="__main__":
    in_dir = r'E:\WorkSpaceSkyMap\MRSAC\image\origin'
    out_dir = r"E:\WorkSpaceSkyMap\MRSAC\image\origin_RGB"
    os.makedirs(out_dir, exist_ok=True)

    for fp in glob.glob(os.path.join(in_dir,'*.tif')):
        fn = os.path.basename(fp)
        fp_img_in = f'E:\WorkSpaceSkyMap\MRSAC\image\origin\{fn}'
        fp_out = os.path.join(out_dir, fn)
        list_idx_band_change = [3,2,1,4]
        change_idx_band(fp_out, fp_img_in, list_idx_band_change)
    