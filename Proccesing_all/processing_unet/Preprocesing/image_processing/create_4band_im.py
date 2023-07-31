import gdal
import numpy as np
from osgeo import gdal, gdalconst, ogr, osr
import matplotlib.pyplot as plt

ds1 = gdal.Open(r'D:\data_source\ESRI\withdsmdtm_3band\ESRI_GD_image1.tif')
data1 = ds1.ReadAsArray()
rgb_img = np.array(data1)

# proj = osr.SpatialReference(wkt=dataset.GetProjection())
# # epsr2 = (proj.GetAttrValue('AUTHORITY',1))

print(rgb_img.shape)

ds2 = gdal.Open(r'D:\data_source\ESRI\withdsmdtm\ESRI_GD_image1_dsm.tif')
data2 = ds2.ReadAsArray()
print(data2.shape)

ds3 = gdal.Open(r'D:\data_source\ESRI\withdsmdtm\ESRI_GD_image1_dtm.tif')
data3 = ds3.ReadAsArray()
print(data3.shape)

four_band_img = []
for i in range(3):
    four_band_img.append(rgb_img[i])
x = np.array(data2[0:8407,0:5622]-data3[0:8407,0:5622])
plt.imshow(x)
plt.show()
four_band_img.append(x)
four_band_img = np.array(four_band_img)

driver = gdal.GetDriverByName("GTiff")
dst_ds = driver.Create('./ESRI_GD_image1_4band.tif',5622,8407,4,gdal.GDT_Float32)#gdal.GDT_Byte/GDT_UInt16
for i in range(1,5):
    dst_ds.GetRasterBand(i).WriteArray(four_band_img[i-1])
    dst_ds.GetRasterBand(i).ComputeStatistics(False)

dst_ds.SetProjection(ds1.GetProjection())
dst_ds.SetGeoTransform(ds1.GetGeoTransform())
dst_ds.FlushCache()