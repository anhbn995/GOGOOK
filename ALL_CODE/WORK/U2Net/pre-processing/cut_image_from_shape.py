# import rasterio
# import rasterio.mask
# import geopandas as gp

# image_path = '/mnt/Nam/public/farm_maxar/10300100A86F9700-visual.tif'
# shape_path = '/mnt/Nam/public/farm_maxar/predict/10300100A86F9700-visual/box.shp'

# with rasterio.open(image_path, mode='r+') as src:
#     projstr = src.crs.to_string()
    
# bound_shp = gp.read_file(shape_path)
# bound_shp = bound_shp.to_crs(projstr)

# for index, row_bound in bound_shp.iterrows():
#     try:
#         geoms = row_bound.geometry
#         img_cut = "/mnt/Nam/public/farm_maxar/data/image_{}.tif".format(index+1)
#         with rasterio.open(image_path) as src:
#             out_image, out_transform = rasterio.mask.mask(src, [geoms], crop=True)
#             print(out_image.shape)
#             out_meta = src.meta
#         out_meta.update({"driver": "GTiff",
#                 "height": out_image.shape[1],
#                 "width": out_image.shape[2],
#                 "transform": out_transform})
#         with rasterio.open(img_cut, "w", **out_meta, compress='lzw') as dest:
#             dest.write(out_image)
#     except:
#         pass

import rasterio, glob, os
import rasterio.mask
import geopandas as gp

image_path = '/mnt/Nam/public/hanoi_sen2/data/image_z18/box_2.tif'
shape_path = '/mnt/Nam/public/hanoi_sen2/box.shp'

with rasterio.open(image_path, mode='r+') as src:
    projstr = src.crs.to_string()
    
bound_shp = gp.read_file(shape_path)
bound_shp = bound_shp.to_crs(projstr)
pathss = glob.glob('/mnt/Nam/public/hanoi_sen2/data/image_z18/*.tif')
for path in pathss:
    for index, row_bound in bound_shp.iterrows():
        try:
            geoms = row_bound.geometry
            img_cut = "/mnt/Nam/public/hanoi_sen2/data/data_z18/"+ os.path.basename(path)
            with rasterio.open(path) as src:
                out_image, out_transform = rasterio.mask.mask(src, [geoms], crop=True)
                print(out_image.shape)
                out_meta = src.meta
            out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})
            with rasterio.open(img_cut, "w", **out_meta, compress='lzw') as dest:
                dest.write(out_image)
        except:
            pass