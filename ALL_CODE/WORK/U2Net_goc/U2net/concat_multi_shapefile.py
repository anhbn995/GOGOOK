import time, cv2
import rasterio.mask
import numpy as np
import pandas as pd
import glob, os
import rasterio
import geopandas as gp
from cythonn import helloworld, Vectorization
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects

for path in glob.glob('/mnt/data/public/farm-bing18/Rs_Cloud_remove/Wajo_result_single/*.shp'):
    name_img = os.path.basename(path).replace('.shp','')
    path_ggmap = '/mnt/data/download_image_tool/result_wajo/{}/*.geojson'.format(name_img)
    path_bing = '/mnt/data/public/farm-bing18/Bingmaps_wajo_predict/05_01/model_u2net/mask/{}.geojson'.format(name_img)
    img = '/mnt/data/public/farm-bing18/Bingmaps_wajo_predict/05_01/model_u2net/mask/{}.tif'.format(name_img)
    save_path = '/mnt/data/Nam_work_space/30_12/Wajo/{}.geojson'.format(name_img)
    print(path)
    try:
        b = []
        end = time.time()
        for i in glob.glob(path_ggmap):
            a = gp.read_file(i)
            b.append(a)
        b = gp.GeoDataFrame(pd.concat(b, ignore_index=True) )
        a = gp.read_file(path_bing)
        cloud_boundary = gp.read_file(path)

        a['ida'] = list(range(len(a)))
        b['idb'] = list(range(len(b)))
        cloud_boundary['idc'] = list(range(len(cloud_boundary)))
        b = b.to_crs("EPSG:4326")
        cloud_boundary = cloud_boundary[['geometry', 'idc']]

        b_c = gp.overlay(b, cloud_boundary, how='intersection')
        a_c = gp.overlay(a, cloud_boundary, how='intersection')

        # a_c['area_total'] = a.iloc[a_c['ida']].area.reset_index()[0]
        # a_c['area_intersection'] = a_c.area
        # heso= a_c['area_intersection']/a_c['area_total']
        # a_c['heso'] = heso
        # a_c = a_c[a_c['heso']>0.95]

        df1 = b.iloc[b_c['idb']]
        df2 = a.iloc[a['ida'].drop(a_c['ida'])]
        df = df2.append(df1)
        gdf = gp.GeoDataFrame(df['geometry'], geometry='geometry', crs='EPSG:4326')
        df_difference = gp.overlay(a, gdf, how='difference')

        x = gdf.append(df_difference)
        gdf = gp.GeoDataFrame(x['geometry'], geometry='geometry', crs='EPSG:4326')
        print(time.time()-end)
        
        with rasterio.open(img) as src:
            height = src.height
            width = src.width
            transform = src.transform
            out_meta = src.meta
            projstr = src.crs.to_string()
        # out_meta.update({"count": 1, "dtype": 'uint8', 'nodata': 0})

        # gdf = gdf.to_crs(projstr)
        # gdf = gdf[gdf['geometry'].type=='Polygon']

        mask = rasterio.features.geometry_mask(gdf['geometry'].boundary, (height, width), transform, invert=True, all_touched=True).astype('uint8')
        # mask = mask & mask_nodata
        # label = img.replace('.tif', '_mask.tif')
        # print(label)
        # with rasterio.open(label, 'w', compress='lzw', **out_meta) as ras:
        #     ras.write(mask[np.newaxis, :, :])
            
        # with rasterio.open('/mnt/data/Nam_work_space/30_12/model_farm/mask/tile_z11_1660_978_mask.tif') as inds:
        #     data = inds.read()
        #     out_meta = inds.meta
        #     transform = inds.transform
        #     projstr = inds.crs.to_string()

        end = time.time()
        kernel = np.ones((3,3),np.uint8)
        img = cv2.dilate(mask,kernel,iterations = 1)
        img = remove_small_holes(img.astype(bool), area_threshold=66)
        img = skeletonize(img)
        print(time.time()-end)
        print(mask.shape)
        # out_img_img = '/mnt/data/Nam_work_space/30_12/skeleton.tif'
        # with rasterio.open(out_img_img, "w", **out_meta, compress='lzw') as dest:
        #     dest.write(skeleton[np.newaxis,...].astype(np.uint8))
            
        # image = '/mnt/data/Nam_work_space/30_12/skeleton.tif'
        # with rasterio.open(image) as inds:
        #     img = inds.read()[0]
        #     transform = inds.transform
        #     projstr = inds.crs.to_string()
        end = time.time()
        Vectorization.save_polygon(np.pad(img, pad_width=1).astype(np.intc), 3,5,transform, projstr, save_path)
        print(time.time()-end)
    except:
        pass