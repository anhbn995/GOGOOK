import rasterio.mask
import rasterio
import numpy as np
import glob, os
import geopandas as gp


"""
INPUT:
    image: path image. VD: $PATH_IMAGE/xxx.tif
    shp: path shapefile
OUTPUT:
    out_label: output image mask. VD: $PATH_IMAGE/xxx_mask.tif

Note:
    "image", "out_label" in the same folder, "image" name is xxx then "out_label" name is xxx_mask
"""

list_fp_img = glob.glob(os.path.join(r'./raw_final', "*.tif"))
for image in list_fp_img:
    # image = '/media/skymap/Learnning/public/change_dubai/Stack_change_Dubai.tif'
    for img in glob.glob(image):
        with rasterio.open(img, mode='r+') as src:
            projstr = src.crs.to_string()
            print(projstr)
        shp = os.path.join(r'./raw_final',os.path.basename(img).replace('tif','shp'))
        # shp = "/home/skm/SKM/WORK/Demo_Kuwait/Kuwait_Planet_project/Label/open_land/label/v2/open_land_32638_fix.shp"
        bound_shp = gp.read_file(shp)
        # bound_shp = bound_shp[bound_shp['geometry'].type=='LineString']
        # bound_shp = bound_shp[bound_shp['geometry'].type=='Polygon']
        bound_shp = bound_shp.to_crs(projstr)

        # for img in glob.glob('/mnt/Nam/public/hanoi_sen2/data/data_z18/*.tif'):
        with rasterio.open(img) as src:
            height = src.height
            width = src.width
            src_transform = src.transform
            out_meta = src.meta
            # mask_nodata = np.ones([height, width], dtype=np.uint8)
            # for i in range(src.count):
            #     mask_nodata = mask_nodata & src.read_masks(i+1)
        out_meta.update({"count": 1, "dtype": 'uint8', 'nodata': 0})

        mask = rasterio.features.geometry_mask(bound_shp['geometry'], (height, width), src_transform, invert=True, all_touched=True).astype('uint8')
        # print(np.unique(mask))
        # mask = mask & mask_nodata
        out_label = img.replace('.tif', '_mask.tif')
        print(out_label)
        with rasterio.open(out_label, 'w', compress='lzw', **out_meta) as ras:
            ras.write(mask[np.newaxis, :, :])



# image = '/mnt/Nam/bairac/classification_data/data_train/*.tif'
# for img in glob.glob(image):
#     with rasterio.open(img, mode='r+') as src:
#         projstr = src.crs.to_string()
#         height = src.height
#         width = src.width
#         src_transform = src.transform
#         out_meta = src.meta
    
#     shp = '/mnt/Nam/bairac/classification_data/landfill_training/'+ os.path.basename(img).replace('tif','shp')
#     bound_shp = gp.read_file(shp)
#     bound_shp = bound_shp.to_crs(projstr)

#     src1 = []
#     src2 = []
#     for i in range(len(bound_shp)):
#         if bound_shp.iloc[i]['id'] == 1:
#             src1.append(bound_shp.iloc[i]['geometry'])
#         else:
#             src2.append(bound_shp.iloc[i]['geometry'])

#     mask_paddy = rasterio.features.geometry_mask(src1, (height, width), src_transform,invert=True, all_touched=True).astype("uint8")
#     mask_background = rasterio.features.geometry_mask(src2, (height, width), src_transform,invert=True, all_touched=True).astype("uint8")
#     mask = mask_paddy+2*mask_background

#     out_meta.update({"count": 1, "dtype": 'uint8', 'nodata': 0})
#     label = img.replace('.tif', '_mask.tif')
#     print(label)
#     with rasterio.open(label, 'w', compress='lzw', **out_meta) as ras:
#         ras.write(mask[np.newaxis, :, :])