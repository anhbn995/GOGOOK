import rasterio
import numpy as np
import glob, os
from itertools import product
from rasterio import windows
import pandas as pd
from shapely.geometry import Polygon, box, mapping, Point, LineString
from shapely.strtree import STRtree
import geopandas as gp
from tqdm import tqdm


def xxx(a):
    return len(a.exterior.coords[:])

def convert(a):
    return ((np.array(a.exterior.coords[:]) - np.array((transform[2], transform[5])))/ np.array((transform[0], transform[4]))).astype(np.uint16)[:4]

def get_tiles(ds, width, height, stride):
    ncols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, ncols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)
    offset = []
    for col_off, row_off in offsets:
        if row_off + width > nrows:
            row_off = nrows - width
        if  col_off + height > ncols:
            col_off = ncols - height
        offset.append((col_off, row_off))
    offset = set(offset)
    for col_off, row_off in tqdm(offset): 
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform
        
txt_path = '/mnt/Nam/official_model/RotationDetection-main/dataloader/data_test/labelTxt/'
image_path = '/mnt/Nam/official_model/RotationDetection-main/dataloader/data_test/Annotations/'
if not os.path.exists(txt_path):
    os.makedirs(txt_path)
if not os.path.exists(image_path):
    os.makedirs(image_path)
output_box = 'box_{}'

img = '/mnt/Nam/xxx_xxx/DaNang_23_07_2009_LowAccuracy.tif'
shp_path = '/mnt/Nam/xxx_xxx/ShipDetection/Ship.shp'

idx = 0
with rasterio.open(img) as inds:
    _width, _height = 512, 512
    meta = inds.meta.copy()
    out_crs = inds.crs.to_string()
    my_geoseries = gp.read_file(shp_path)
    my_geoseries = my_geoseries.to_crs(out_crs)
    my_geoseries = my_geoseries[my_geoseries['geometry'].type == 'Polygon']
    b=STRtree(my_geoseries['geometry'])

    for window, transform in get_tiles(inds, _width, _height, 400):
        meta['transform'] = transform
        meta['width'], meta['height'] = window.width, window.height
        min = transform * (0, 0)
        max = transform * (_width, _height)
        boxx = box(*np.minimum(min,max), *np.maximum(min,max))
        geometry = [o.intersection(boxx) for o in b.query(boxx) if o.intersects(boxx)]
        if len(geometry) >0:
            outpath_image = os.path.join(image_path, output_box.format('{0:003d}'.format(idx)))+ '.tif'
            out_txt = os.path.join(txt_path, output_box.format('{0:003d}'.format(idx)))+ '.txt'
            
            data_fame = pd.DataFrame(geometry, columns=['geometry'])
            gdf = gp.GeoDataFrame(data_fame, geometry='geometry',crs=out_crs)
            check = gdf['geometry'].apply(xxx)
            gdf = gdf[check==5]
            gdf = gdf['geometry'].apply(convert)
            # A = [" ".join(str(e) for e in i.reshape(-1)) + " car" + " 1" + " \n" for i in gdf]
            A = [" ".join(str(e) for e in i.reshape(-1)) + " car" + " 1" + " \n" for i in gdf if len(np.unique([tuple(row) for row in i], axis=0))==4]
            if not A: print(gdf)
            if A:
                with open(out_txt, 'a') as file:
                    file.writelines(A)
                with rasterio.open(outpath_image, "w", **meta, compress='lzw') as dest:
                    dest.write(inds.read(window=window))
                idx+=1