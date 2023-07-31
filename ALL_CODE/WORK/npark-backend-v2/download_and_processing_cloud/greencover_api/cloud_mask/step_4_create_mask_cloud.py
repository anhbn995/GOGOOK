import os
import glob
import rasterio
import numpy as np

from download_and_processing_cloud.greencover_api.cloud_mask.utils import convert_obj
from download_and_processing_cloud.greencover_api.cloud_mask.raster_to_vector import raster_to_vector

def create_mask_cloud(out_cloud_mask, img_base_for_cloud, shp_folder, name):
    if not os.path.exists(shp_folder):
        os.mkdir(shp_folder)
    shp_file = os.path.join(shp_folder, name+'.shp')
    mask = rasterio.open(out_cloud_mask).read_masks(1)
    mask = convert_obj(mask)

    mask1 = rasterio.open(img_base_for_cloud).read_masks(1)
    mask1 = convert_obj(mask1)

    mask2 = mask - mask1
    mask2[mask2!=1]=0

    with rasterio.open(out_cloud_mask) as src:
            transform1 = src.transform
            w,h = src.width,src.height
            crs = src.crs
    tif_file = out_cloud_mask.replace('.tif', '_mask_cloud.tif')
    new_dataset = rasterio.open(tif_file, 'w', driver='GTiff',
                                height = h, width = w,
                                count=1, dtype="uint8",
                                crs=crs,
                                transform=transform1,
                                compress='lzw')
    new_dataset.write(mask2.astype(np.uint8),1)
    new_dataset.close()
    print("zzzzzzzzzzzzzzzzzzzzzzzz", shp_file)
    raster_to_vector(tif_file, shp_file)
    return tif_file, shp_file