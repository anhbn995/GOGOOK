import rasterio
import numpy as np


path_mask = r"/mnt/66A8E45DA8E42CED/Malaysia_Features_Extraction/Training_datasets_BG31141/Data_train/Water/mask/BG31141.tif"
out_path = r"/mnt/66A8E45DA8E42CED/Malaysia_Features_Extraction/Training_datasets_BG31141/Data_train/Water/mask_unet50/BG31141.tif"
src =  rasterio.open(path_mask)
h = src.height
w = src.width
img = src.read()

a = img[0]*255
b = (1-img[0])*255

img_new = np.asarray([a,b], dtype='uint8')


with rasterio.open(
    out_path,
    'w',
    driver='GTiff',
    height=h,
    width=w,
    count=2,
    dtype=img_new.dtype,
    crs=src.crs,
    transform=src.transform,
) as dst:
    dst.write(img_new)
    