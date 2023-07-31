import cv2
import rasterio
import numpy as np
import skimage.morphology

def dilation_obj(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask_base = cv2.dilate(img,kernel,iterations = 1)
    return mask_base

def remove_small_items(img, threshhlod_rm_holes= 256, threshhold_rm_obj=100):
    img_tmp = np.asarray(img, dtype=np.bool)
    img_tmp = skimage.morphology.remove_small_holes(img_tmp, area_threshold=threshhlod_rm_holes)
    img_tmp = skimage.morphology.remove_small_objects(img_tmp, min_size=threshhold_rm_obj)
    return img_tmp

def write_image(image_path, img):
    with rasterio.open(image_path) as src:
        out_meta = src.meta
        crs = src.crs
        tr = src.transform
        height = src.height
        width = src.width
    print("Write image...")
    result_path = image_path.replace('.tif', '_result.tif')

    with rasterio.Env():
        # Write an array as a raster band to a new 8-bit file. For
        # the new file's profile, we start with the profile of the source
        profile = out_meta

        # And then change the band count to 1, set the
        # dtype to uint8, and specify LZW compression.
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw')

    with rasterio.open(result_path, 'w', **profile) as dst:
        dst.write(img.astype(np.uint8),1)
        
#     new_dataset = rasterio.open(result_path, 'w', driver='GTiff',
#                                 height = height, width = width,
#                                 count = 1, dtype = "uint8",
#                                 crs = crs,
#                                 transform = tr,
#                                 # nodata = 0,
#                                 compress='lzw')
    
#     new_dataset.write(img, 1)
#     new_dataset.close()
    return result_path

