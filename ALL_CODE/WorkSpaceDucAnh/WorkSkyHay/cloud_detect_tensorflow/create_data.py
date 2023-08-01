from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import rasterio
import geopandas
import shapely
from rasterio.plot import show
from rasterio.mask import raster_geometry_mask
from rasterio.plot import reshape_as_image
import fiona
import pyproj

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.cm as cm
from PIL import Image
import math
import os


data_locate = r'./raw_final/add_more.tif'
shp_locate = r'./raw_final/add_more.shp'

ds = rasterio.open(data_locate,'r+')
with rasterio.open(data_locate) as src:
    out_meta = src.meta
out_meta.update({"count": 1, "dtype": 'uint8', 'nodata': 0})
data = ds.read()

with fiona.open(shp_locate, "r") as shapefile: #
    meta = shapefile.meta
    shapes = [feature["geometry"] for feature in shapefile]

with rasterio.open(data_locate) as src:
    height = src.height
    width = src.width
    src_transform = src.transform
    out_meta = src.meta
    img_filter_mask = src.read_masks(1)
    out_meta.update({"count": 1, "dtype": 'uint8', 'nodata': 0})

    mask = rasterio.features.geometry_mask(shapes, (height, width), src_transform, invert=True, all_touched=True).astype('uint8')
    mask[img_filter_mask==0]=0

data = reshape_as_image(data)
height=300
width=300

neg = 0
pos = 0
isContain = 0
for i in range(1000+300//2, data.shape[0]-2000//2, height//3):
    for j in range(800+300//2, data.shape[1]-2000//2, width//3):
        
        
        train = Image.fromarray(data[i-height//2:i+height//2,j-width//2:j+width//2,0:3].astype(np.uint8))        
        msk = Image.fromarray(255*mask[i-height//2:i+height//2,j-width//2:j+width//2].astype(np.uint8))

        if np.array(msk).sum() == 0:
            # trainImg = os.path.join(img_path,'trainImg_'+str(neg+7902+6574+6727+7243+7955)+'.png')
            # trainMsk = os.path.join(mask_path, 'trainMsk_'+str(neg+7902+6574+6727+7243+7955)+'.bmp')
            neg+=1
        else:            
            # Positive class
            trainImg = os.path.join(img_path2,'trainImg_'+str(pos+481+595+549+337+307)+'.png')
            trainMsk = os.path.join(mask_path2, 'trainMsk_'+str(pos+481+595+549+337+307)+'.bmp')
            pos+=1
            train.save(trainImg)
            msk.save(trainMsk)
        
        
        if (neg+pos)%500==0: print('Image', (neg+pos),'generated!')

print('Finished: Created',(neg+pos),'image')
print("INFO: \nPositive class =",pos, "-- Negative class =",neg)
print("Postive/Negative rate =", pos/neg)
