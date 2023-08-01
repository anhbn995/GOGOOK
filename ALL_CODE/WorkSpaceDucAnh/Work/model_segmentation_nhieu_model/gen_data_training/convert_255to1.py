import rasterio
import numpy as np
import glob, os

dir_img = r"/home/skm/SKM_OLD/ZZ_ZZ/cloud_shadow/create_data_train_uint8/img_origin/zmask_255"
dir_out = r"/home/skm/SKM_OLD/ZZ_ZZ/cloud_shadow/create_data_train_uint8/img_origin/zmask_01"
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
list_fp = glob.glob(dir_img + "/*.tif")
for fp_img in list_fp:
    with rasterio.open(fp_img) as src:
        meta = src.meta
        img = src.read()
        img[img==255]=1
    # fp_img_out = os.path.join(dir_out, os.path.basename(fp_img).replace('.tif', '_mask.tif'))
    fp_img_out = os.path.join(dir_out, os.path.basename(fp_img))
    print(fp_img_out)
    with rasterio.open(fp_img_out, 'w', **meta) as dst:
        dst.write(img)