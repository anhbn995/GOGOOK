import rasterio
import numpy as np
from math import sqrt
import timeit, time, os, glob
from cythonn import helloworld, Vectorization
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects

path_img = '/mnt/data/public/farm-bing18/Bingmaps_wajo_predict/05_01/model_u2net/mask/*.tif'
# for img_path in os.listdir(path_img):
for img_path in glob.glob(path_img):
    # out_img = '/mnt/data/public/farm-bing18/BingMaps_Kediri_predict/30_12/model_farm/skeleton/'+img_path
    # if not os.path.exists(img_path):
    with rasterio.open(img_path) as f:
        data = f.read()
        out_meta = f.meta
        transform = f.transform
        projstr = f.crs.to_string()
    star = time.time()
    data = remove_small_holes(data.astype(bool), area_threshold=77)
    data = remove_small_objects(data, min_size=77)
    skeleton = skeletonize(data.astype(np.uint8))
    # print(out_img)
    # with rasterio.open(out_img, "w", **out_meta, compress='lzw') as dest:
    #     dest.write(skeleton)
        
    # path_img = "/mnt/data/Nam_work_space/30_12/aaa/"
    # for img_path in os.listdir(path_img):
    #     image = path_img+img_path
    #     with rasterio.open(image) as inds:
    # img = inds.read()[0]
    # transform = inds.transform
    # projstr = inds.crs.to_string()
    # save_path = "/mnt/data/Nam_work_space/30_12/aaa/" + img_path.split('.')[0]+'.geojson'
    save_path = img_path.replace('.tif','.geojson')
    # if not os.path.exists(save_path):
    try:
        print(save_path)
        test = Vectorization.save_polygon(np.pad(skeleton[0], pad_width=1).astype(np.intc), 3,5,transform, projstr, save_path)
        print(time.time()-star)
    except:
        print(100*'-')
        print(save_path)
        print(100*'-')

# import time, cv2
# import rasterio.mask
# import numpy as np
# import pandas as pd
# import glob, os
# import rasterio
# import geopandas as gp
# from cythonn import helloworld, Vectorization
# from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects

# for path in glob.glob('/mnt/data/public/farm-bing18/Rs_Cloud_remove/kediri_result_single/tile_z11_1662_980.shp'):
#     name_img = os.path.basename(path).replace('.shp','')
#     path_ggmap = '/mnt/data/download_image_tool/result_kediri/{}/*.geojson'.format(name_img)
#     path_bing = '/mnt/data/farmmmm/tile_z11_1662_980.geojson'
#     img = '/mnt/data/farmmmm/tile_z11_1662_980.tif'
#     save_path = '/mnt/data/farmmmm/a/{}.geojson'.format(name_img)
#     print(path)
#     try:
#         b = []
#         end = time.time()
#         for i in glob.glob(path_ggmap):
#             a = gp.read_file(i)
#             b.append(a)
#         b = gp.GeoDataFrame(pd.concat(b, ignore_index=True) )
#         a = gp.read_file(path_bing)
#         cloud_boundary = gp.read_file(path)

#         a['ida'] = list(range(len(a)))
#         b['idb'] = list(range(len(b)))
#         cloud_boundary['idc'] = list(range(len(cloud_boundary)))
#         b = b.to_crs("EPSG:4326")
#         cloud_boundary = cloud_boundary[['geometry', 'idc']]

#         b_c = gp.overlay(b, cloud_boundary, how='intersection')
#         a_c = gp.overlay(a, cloud_boundary, how='intersection')

#         df1 = b.iloc[b_c['idb']]
#         df2 = a.iloc[a['ida'].drop(a_c['ida'])]
#         df = df2.append(df1)
#         gdf = gp.GeoDataFrame(df['geometry'], geometry='geometry', crs='EPSG:4326')
#         df_difference = gp.overlay(a, gdf, how='difference')

#         x = gdf.append(df_difference)
#         gdf = gp.GeoDataFrame(x['geometry'], geometry='geometry', crs='EPSG:4326')
#         print(time.time()-end)

#         with rasterio.open(img) as src:
#             height = src.height
#             width = src.width
#             transform = src.transform
#             out_meta = src.meta
#             projstr = src.crs.to_string()

#         mask = rasterio.features.geometry_mask(gdf['geometry'].boundary, (height, width), transform, invert=True, all_touched=True).astype('uint8')
#         end = time.time()
#         kernel = np.ones((3,3),np.uint8)
#         img = cv2.dilate(mask,kernel,iterations = 1)
#         img = remove_small_holes(img.astype(bool), area_threshold=66)
#         img = skeletonize(img)
#         print(time.time()-end)
#         print(mask.shape)
#         end = time.time()
#         Vectorization.save_polygon(np.pad(img, pad_width=1).astype(np.intc), 3,5,transform, projstr, save_path)
#         print(time.time()-end)
#     except:
#         pass