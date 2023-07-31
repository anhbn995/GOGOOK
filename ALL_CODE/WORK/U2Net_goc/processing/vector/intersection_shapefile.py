import os
import geopandas as gp
import pandas as pd
import numpy as np
import copy
import itertools
from shapely.geometry import shape, mapping
import rasterio
import rasterio.features
import time

def remove_polygon(a):
    if a.type == 'Polygon':
        if a.area > 3e-9:
            return [a]
    else:
        nam = [i for i in a if i.area>3e-9]
        if nam:
            return nam
        

def remove_lines(a):
    if a.type == 'Polygon':
        if a.area > 3e-9:
            return [a.boundary]
    else:
        nam = [i.boundary for i in a if i.area>3e-9]
        if nam:
            return nam

path_img = '/media/skymap/Learnning/public/farm-bing18/Data/predicttttttttttttttt/skeleton_v2/'
for img_path in os.listdir(path_img):
    label = '/media/skymap/Learnning/public/farm-bing18/Data/predicttttttttttttttt/label/'+img_path
    if os.path.exists(label):
        continue
    try:
        a = time.time()
        print(img_path)
        df1 = gp.read_file('/media/skymap/Learnning/public/farm-bing18/Data/predicttttttttttttttt/shapefile_v1/'+img_path.split('.')[0]+'.shp')
        df2 = gp.read_file('/media/skymap/Learnning/public/farm-bing18/Data/predicttttttttttttttt/shapefile_v2/'+img_path.split('.')[0]+'.shp')
#         df3 = gp.read_file('/media/skymap/Learnning/public/farm-bing18/Data/predict_new_farm/shapefilee/'+img_path.split('.')[0]+'.shp')
        gdx1 = df1.to_crs('EPSG:3857')
        gdx2 = df2.to_crs('EPSG:3857')
#         gdx3 = df3.to_crs('EPSG:3857')

        df1 = df1[gdx1['geometry'].area<99999]
        df2 = df2[gdx2['geometry'].area<99999]
#         df3 = df3[gdx3['geometry'].area<99999]

        df1_df2 = gp.overlay(df2, df1, how='difference')
        df1_df2 = df1.append(df1_df2)
        xx = df1_df2['geometry'].apply(remove_lines).reset_index().dropna()
        # xx = df1_df2['geometry'].apply(remove_polygon).reset_index().dropna()
        geogeo = list(itertools.chain(*xx.geometry.to_list()))
        data_fame = pd.DataFrame({'FID': list(np.arange(len(geogeo))), 'geometry':geogeo})
        gdf = gp.GeoDataFrame(data_fame, geometry='geometry', crs='EPSG:4326')

        
#         df = gp.overlay(df3, gdf, how='difference')
#         df = gdf.append(df)
#         xxxx = df['geometry'].apply(remove_lines).reset_index().dropna()
#         geogeogeo = list(itertools.chain(*xxxx.geometry.to_list()))
#         data_famess = pd.DataFrame({'FID': list(np.arange(len(geogeogeo))), 'geometry':geogeogeo})
#         gdf = gp.GeoDataFrame(data_famess, geometry='geometry', crs='EPSG:4326')
        
        
        gdf.to_file('/media/skymap/Learnning/public/farm-bing18/Data/predicttttttttttttttt/maskk/'+img_path.split('.')[0]+'.shp')
        
        img = path_img+img_path
        with rasterio.open(img) as src:
            height = src.height
            width = src.width
            src_transform = src.transform
            out_meta = src.meta
            
        mask = rasterio.features.geometry_mask(gdf['geometry'], (height, width), src_transform,invert=True, all_touched=True).astype(np.uint8)
        out_meta.update({"count": 1, 'nodata':0})
        with rasterio.open(label, 'w', compress='lzw', **out_meta) as ras:
            ras.write(mask[np.newaxis, :, :])
        print(time.time()-a)
    except Exception as e:
        print(50*'-')
        print(e)
        print(50*'-')