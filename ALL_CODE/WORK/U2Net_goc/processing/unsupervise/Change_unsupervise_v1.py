# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 09:04:30 2021

@author: AnhHo
"""
import rasterio
import numpy as np
import cv2

base_path=r"/mnt/data/public/changedetection_SAR/pipeline/Raw/Costa Rica S1A Dsc 22-Oct-2021.tif"
image_path=r"/mnt/data/public/changedetection_SAR/pipeline/Raw/Costa Rica S1A  Dsc 24-February-2021.tif"
outputFileName = r"/mnt/data/changedetection_SAR/image/Costa Rica S1A result 24Feb_22Oct.tif"

def get_quantile_schema(img):
    qt_scheme = []
    with rasterio.open(img) as r:
        num_band = r.count
        for chanel in range(1,num_band+1):
            data = r.read(chanel).astype(np.float16)
            data[data==0] = np.nan
            qt_scheme.append({
                'p2': np.nanpercentile(data, 2),
                'p98': np.nanpercentile(data, 98),
            })
    print(qt_scheme)
    return qt_scheme

def stretch_image(data,qt_scheme,mask,profile):
        ids = range(len(data))
        window_image = data
        cut_shape = np.shape(window_image)
        new_image = np.zeros((cut_shape[1], cut_shape[2], len(ids)), dtype=np.uint8)
        for i in ids:
            band = window_image[i]
            try:
                if profile['dtype'] == 'uint8':
                    cut_nor = band.astype(int)
                else:
                    band_qt = qt_scheme[i]
                    cut_nor = np.interp(band, (band_qt.get('p2'), band_qt.get('p98')), (1, 255)).astype(int)
            except Exception:
                cut_nor = band.astype(int)
            band[~mask[i]] = 0
            new_image[..., i] = cut_nor

        result = new_image/255.0
        return result, mask
    
with rasterio.open(base_path) as src1:
#    img1 = src1.read().transpose(1,2,0).astype(np.float16)
    img1 = src1.read()
    profile1 = src1.profile
    mask1 = src1.read_masks()
    transform1 = src1.transform
    w,h = src1.width, src1.height
    crs1 = src1.crs
with rasterio.open(image_path) as src2:
    img2 = src2.read()
    profile2 = src2.profile
    mask2 = src2.read_masks()
    
qt_scheme1 = get_quantile_schema(base_path)
qt_scheme2 = get_quantile_schema(image_path)
img1_streth,mask = stretch_image(img1,qt_scheme1,mask1,profile1)
img2_streth,mask = stretch_image(img2,qt_scheme2,mask2,profile2)
img_change = (img1_streth - img2_streth)

from numpy import linalg as LA
#result = LA.norm(np.expand_dims(img_change[:,:,1], axis=-1), axis=-1)
result = LA.norm(img_change, axis=-1)
import matplotlib.pyplot as plt
#plt.imshow(result, cmap='gray')
# plt.hist(result.flatten(),bins=1000)
# plt.show()
result_cal = result.copy()
result_cal[result_cal==0] = np.nan
a = np.nanpercentile(result_cal,95)
result1 = (result>=a).astype(np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img = cv2.morphologyEx(result1, cv2.MORPH_CLOSE, kernel)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel3)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel3)


new_dataset = rasterio.open(outputFileName, 'w', driver='GTiff',
                            height = h, width = w,
                            count=1, dtype="uint8",
                            crs=crs1,
                            transform=transform1,
                            compress='lzw',
                            nodata=0)

new_dataset.write(img,1)
new_dataset.close()