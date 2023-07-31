# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:44:28 2018

@author: Huett
"""


import numpy as np
import rasterio
import pandas as pd
import rasterio.features
import shapely.ops
import shapely.geometry
import shapely.wkt
#from logging import getLogger, Formatter, StreamHandler, INFO




def mask_to_poly(mask, min_polygon_area_th=30, threshold = 0.5):

    print("bat dau tinh shape")
    shapes = rasterio.features.shapes(mask.astype(np.int16), mask > 0)
    print("tinh shape xong & tim polygon")
    mp = shapely.ops.cascaded_union(
        shapely.geometry.MultiPolygon([
            shapely.geometry.shape(shape)
            for shape, value in shapes
        ]))
    
    print("luu du lieu ra data frame")
    if isinstance(mp, shapely.geometry.Polygon):
        df = pd.DataFrame({
            'area_size': [mp.area],
            'poly': [mp],
        })
    else:
        df = pd.DataFrame({
            'area_size': [p.area for p in mp],
            'poly': [p for p in mp],
        })
    print(df.area_size)
    df = df[df.area_size > min_polygon_area_th]
    print ("sau khi bo vung be")
    print(df.area_size)

#    df = df[df.area_size > min_polygon_area_th].sort_values(
#        by='area_size', ascending=False)
#    df.loc[:, 'wkt'] = df.poly.apply(lambda x: shapely.wkt.dumps(
#        x, rounding_precision=0))
#    df.loc[:, 'bid'] = list(range(1, len(df) + 1))
#    df.loc[:, 'area_ratio'] = df.area_size / df.area_size.max()
    return df

