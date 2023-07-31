import glob, os
from importlib import machinery
import rasterio
import numpy as np
from tqdm import tqdm

def dich_his(fp_in, value_min_can_dich, out_fp):
    # value_min_can_dich = 55
    # fp = r"/home/skm/SKM16/Work/OpenLand/all_tif_sap_xep/from130_255/20220810_073235_ssc1_u0001_visual.tif"
    with rasterio.open(fp_in) as src:
        meta = src.meta
        # mask = src.read_masks()
        # index_nodata = np.where(mask==0)
        # del mask
        img = src.read()

    img[img==0] = 255
    # min = np.nanpercentile(img[0], 2)
    # min = np.nanpercentile(img[0], 5)
    
    # chuyen except
    min = 120
    # out_fp = out_fp.replace('.tif', f'{str(min)}.tif')


    print(min)
    value_dich = int(min - value_min_can_dich)
    img[img < value_dich] = value_dich + 1
    img = img - value_dich
    # img[index_nodata] = 0
    # out_fp = f"/home/skm/SKM16/Work/OpenLand/all_tif_sap_xep/from130_255/out_fp/20220810_073235_ssc1_u0001_visual_{str(value_dich)}_2.tif"
    with rasterio.open(out_fp,'w',**meta) as dst:
        dst.write(img)

if __name__ == "__main__":
    # dir_name = r"/home/skm/SKM16/Work/OpenLand/all_tif/rm_thieu/20220818_073620_ssc1_u0001_visual.tif"
    # dir_out = r"/home/skm/SKM16/Work/OpenLand/all_tif/rm_thieu/chuyen_ve_anhKuwait1/20220818_073620_ssc1_u0001_visual.tif"
    # value_min_can_dich = 55
    # dich_his(dir_name, value_min_can_dich, dir_out)

    dir_name = r"/home/skm/SKM16/Work/OpenLand/all_tif_sap_xep/Miss/chua_run/120"
    dir_out = r"/home/skm/SKM16/Work/OpenLand/all_tif_sap_xep/Miss/chua_run/120/chuyen_ve_anhKuwait"
    os.makedirs(dir_out, exist_ok=True)
    value_min_can_dich = 55

    list_fp = glob.glob(os.path.join(dir_name,'*.tif'))
    for fp_in in tqdm(list_fp):
        out_fp = os.path.join(dir_out, os.path.basename(fp_in))
        dich_his(fp_in, value_min_can_dich, out_fp)

