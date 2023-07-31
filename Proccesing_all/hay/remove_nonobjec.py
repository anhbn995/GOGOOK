import cv2
import rasterio
import numpy as np


def arr2raster(path_out, bands, height, width, tr, dtype="uint8",coordinate=None,projstr=None):
    num_band = len(bands)
    if coordinate!= None:
        crs = rasterio.crs.CRS.from_epsg(coordinate)
    else:
        crs = rasterio.crs.CRS.from_string(projstr)
    new_dataset = rasterio.open(path_out, 'w', driver='GTiff',
                            height = height, width = width,
                            count = num_band, dtype = dtype,
                            crs = crs,
                            transform = tr,
                            nodata = 0,
                            compress='lzw')
    if num_band == 1:
        new_dataset.write(bands[0], 1)
    else:
        for i in range(num_band):
            new_dataset.write(bands[i],i+1)
    new_dataset.close()


def arr2raster(path_out, bands, height, width, tr, crs, dtype="uint8",coordinate=None,projstr=None):
    num_band = len(bands)
    # if coordinate!= None:
    #     crs = rasterio.crs.CRS.from_epsg(coordinate)
    # else:
    #     crs = rasterio.crs.CRS.from_string(projstr)
    new_dataset = rasterio.open(path_out, 'w', driver='GTiff',
                            height = height, width = width,
                            count = num_band, dtype = dtype,
                            crs = crs,
                            transform = tr,
                            nodata = 0,
                            compress='lzw')
    if num_band == 1:
        new_dataset.write(bands[0], 1)
    else:
        for i in range(num_band):
            new_dataset.write(bands[i],i+1)
    new_dataset.close()

path_remove = r"/home/skm/SKM/WORK/Sinarmas_all/Deading_tree/1_Image_uint16_convert_01_float/Non_Uniformity_float_64_xz/mosaic_16092020_sr.tif"

path_road = r"/home/skm/SKM/WORK/Sinarmas_all/Deading_tree/Model_Uint16_Label_Non_Object/img_01/_predict_model_float_Non_ROAD_float_256_xz/mosaic_16092020_sr.tif"
path_cloud = r"/home/skm/SKM/WORK/Sinarmas_all/Deading_tree/Model_Uint16_Label_Non_Object/img_01/_predict_model_float_Non_CLOUD_float_128_xz/mosaic_16092020_sr.tif"
path_buildUp = r"/home/skm/SKM/WORK/Sinarmas_all/Deading_tree/Model_Uint16_Label_Non_Object/img_01_mask_BUILDUP/mosaic_16092020_sr.tif"




src1 = rasterio.open(path_road)
src2 = rasterio.open(path_cloud)
src3 = rasterio.open(path_buildUp)
src = rasterio.open(path_remove)

arr1 = src1.read()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
arr1 = cv2.dilate(arr1[0],kernel,iterations = 1)
arr1 = np.array([arr1])
arr2 = src2.read()
arr3 = src3.read()
arr = src.read()

index1 = np.where(arr1==255)
index2 = np.where(arr2==255)
index3 = np.where(arr3==255)

arr[index1]=0
arr[index2]=0
arr[index3]=0





path_out = r"/home/skm/SKM/WORK/Sinarmas_all/Deading_tree/1_Image_uint16_convert_01_float/Result_rm_noobject/mosaic_16092020_Non_Uniformity.tif"
arr2raster(path_out, arr, src.height, src.width, src.transform, src.crs, dtype="uint8")
