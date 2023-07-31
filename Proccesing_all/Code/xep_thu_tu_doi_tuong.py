import rasterio
from rasterio.transform import from_origin
import numpy as np
import cv2
def masking(url):
    with rasterio.open(url) as src:
        arr = src.read()[0]
    return (arr == 255).astype(np.uint8)
with rasterio.open(r'/media/khoi/Data1/Gap2/Reuslt_and_Image/Image.tif') as src:
    transform1 = src.transform
    w,h = src.width,src.height
# print(arr.shape)
print(src.res)
# transform = from_origin(362351,144477, 0.6, 0.6)
# print(transform)
crs = rasterio.crs.CRS({"init": "epsg:4326"})
new_dataset = rasterio.open(r'/media/khoi/Data1/Gap2/Reuslt_and_Image/Image_Result_1.tif', 'w', driver='GTiff',
                            height = h, width = w,
                            count=1, dtype="uint8",
                            crs=crs,
                            transform=transform1,
                            compress='lzw')
# print(masking(r"/media/building/building/data_source/tmp/Malaysia-jupem/image_mask/vegetation.tif").shape)
vegetation = masking(r"/media/khoi/Data1/Gap2/Reuslt_and_Image/mask_Vegetation.tif")
buildUp = masking(r"/media/khoi/Data1/Gap2/Reuslt_and_Image/mask_BuildUp.tif")
road = masking(r"/media/khoi/Data1/Gap2/Reuslt_and_Image/mask_Road.tif")
water = masking(r"/media/khoi/Data1/Gap2/Reuslt_and_Image/mask_Water.tif")
data_mask = np.ones((h,w), dtype=np.uint8)

all_lable = cv2.bitwise_or(vegetation,buildUp)
all_lable = cv2.bitwise_or(all_lable,road)
all_lable = cv2.bitwise_or(all_lable,water)
water =  cv2.bitwise_and(all_lable,water)
water_road_intersec = cv2.bitwise_and(road,water)
road = cv2.bitwise_xor(road,water_road_intersec)
water_road_intersec_buildUp = cv2.bitwise_and(buildUp,cv2.bitwise_or(water,road))
buildUp = cv2.bitwise_xor(buildUp,water_road_intersec_buildUp)
water_road_buildUp_intersec_verget = cv2.bitwise_and(vegetation,((buildUp+road+water)>=1).astype(np.uint8))
veget = cv2.bitwise_xor(vegetation,water_road_buildUp_intersec_verget)
data_mask = data_mask + 1*veget + 2*buildUp + 3*road + 4*water


new_dataset.write(data_mask,1)
# new_dataset.write(masking(r"/media/building/building/data_source/tmp/Malaysia-jupem/image_mask/water.tif"),4)   anh ngu day bb chu
# new_dataset.write(masking(r"/media/building/building/data_source/tmp/Malaysia-jupem/image_mask/road.tif"),5)
new_dataset.close()
