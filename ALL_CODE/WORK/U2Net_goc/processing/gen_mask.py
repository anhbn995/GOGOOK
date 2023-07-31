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


# image = '/mnt/Nam/tmp_Nam/pre-processing/road/img/20220404_132910_ssc17_u0001_visual_clip_2.tif'
list_fp_img = glob.glob(os.path.join(r"/home/skymap/big_data/ml_data/DATA_FARM/aus", "*.tif"))
# image = '/mnt/Nam/public/change_new/image_train_8bit_perimage/*.tif'
for image in list_fp_img:
    # image = '/media/skymap/Learnning/public/change_dubai/Stack_change_Dubai.tif'
    for img in glob.glob(image):
        with rasterio.open(img, mode='r+') as src:
            projstr = src.crs.to_string()
            print(projstr)
    shp = os.path.join(r'/home/skymap/big_data/ml_data/DATA_FARM/aus', os.path.basename(img).replace('tif','geojson'))
    # shp = "/mnt/Nam/tmp_Nam/pre-processing/road_multi/img/newdata/marid/shape/polyline.shp"
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
        img_filter_mask = src.read_masks(1)
        # mask_nodata = np.ones([height, width], dtype=np.uint8)
        # for i in range(src.count):
        #     mask_nodata = mask_nodata & src.read_masks(i+1)
    out_meta.update({"count": 1, "dtype": 'uint8', 'nodata': 0})

    mask = rasterio.features.geometry_mask(bound_shp['geometry'], (height, width), src_transform, invert=True, all_touched=True).astype('uint8')
    # print(np.unique(mask))
    # mask = mask & mask_nodata
    out_label = img.replace('.tif', '_mask.tif')
    print(out_label)
    mask[img_filter_mask==0]=0
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