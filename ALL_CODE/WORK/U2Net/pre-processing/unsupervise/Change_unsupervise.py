# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 09:04:30 2021

@author: AnhHo
"""
import rasterio
import numpy as np
import cv2
base_path1=r"Z:\changedetection_SAR\pipeline\Raw\Costa Rica S1A Dsc 10-Oct-2021.tif"
base_path2=r"Z:\changedetection_SAR\pipeline\Raw\Costa Rica S1A Dsc 22-Oct-2021.tif"
image_path1=r"Z:\changedetection_SAR\pipeline\Raw\Costa Rica S1A  Dsc 12-February-2021.tif"
image_path2=r"Z:\changedetection_SAR\pipeline\Raw\Costa Rica S1A  Dsc 24-February-2021.tif"
outputFileName = r"Z:\changedetection_SAR\pipeline\Raw\Costa Rica S1A result Feb_Oct-2021_v8_multi.tif"
def get_quantile_schema(img):
    qt_scheme = []
    # with COGReader(img) as cog:
    #     stats = cog.stats()
    #     for _, value in stats.items():
    #         qt_scheme.append({
    #             'p2': value['percentiles'][0],
    #             'p98': value['percentiles'][1],
    #         })
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
                    # print(123)
                    # self.profile['dtype'] = 'uint8'
                    # nodatamask = (band == np.uint8(0))
            except Exception:
                cut_nor = band.astype(int)
            band[~mask[i]] = 0
            new_image[..., i] = cut_nor

        result = new_image/255.0
        return result, mask
with rasterio.open(base_path1) as src1:
#    img1 = src1.read().transpose(1,2,0).astype(np.float16)
    img1 = src1.read()
    profile1 = src1.profile
    mask1 = src1.read_masks()
    transform1 = src1.transform
    w,h = src1.width,src1.height
    crs1 = src1.crs
with rasterio.open(base_path2) as src2:
    img2 = src2.read()
    profile2 = src2.profile
    mask2 = src2.read_masks()
with rasterio.open(image_path1) as src3:
    img3 = src3.read()
    profile3 = src3.profile
    mask3 = src3.read_masks()
with rasterio.open(image_path2) as src4:
    img4 = src4.read()
    profile4 = src4.profile
    mask4 = src4.read_masks()
qt_scheme1 = get_quantile_schema(base_path1)
qt_scheme2 = get_quantile_schema(base_path2)
qt_scheme3 = get_quantile_schema(image_path1)
qt_scheme4 = get_quantile_schema(image_path2)
img1_streth,mask = stretch_image(img1,qt_scheme1,mask1,profile1).astype(np.float16)
img2_streth,mask = stretch_image(img2,qt_scheme2,mask2,profile2).astype(np.float16)
img3_streth,mask = stretch_image(img3,qt_scheme3,mask3,profile3).astype(np.float16)
img4_streth,mask = stretch_image(img4,qt_scheme4,mask4,profile4).astype(np.float16)

img_change1 = (img1_streth - img3_streth).astype(np.float16)
img_change2 = (img1_streth - img4_streth).astype(np.float16)
img_change3 = (img2_streth - img3_streth).astype(np.float16)
img_change4 = (img2_streth - img4_streth).astype(np.float16)

img_change = np.concatenate([img_change1,img_change2,img_change3,img_change4],axis = -1).astype(np.float16)
from numpy import linalg as LA
#result = LA.norm(np.expand_dims(img_change[:,:,1], axis=-1), axis=-1)
result = LA.norm(img_change, axis=-1)
import matplotlib.pyplot as plt
#plt.imshow(result, cmap='gray')
plt.hist(result.flatten(),bins=1000)
plt.show()
result_cal = result.copy()
result_cal[result_cal==0] = np.nan
a = np.nanpercentile(result_cal,90)
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