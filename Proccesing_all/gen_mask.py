import rasterio.mask
import rasterio
import numpy as np
import glob, os
import geopandas as gp

def create_mask(fp_img, fp_shp):
    with rasterio.open(fp_img, mode='r+') as src:
        projstr = src.crs.to_string()
        print(projstr)
    bound_shp = gp.read_file(fp_shp)
    bound_shp = bound_shp[bound_shp['geometry'].type=='Polygon']
    bound_shp = bound_shp.to_crs(projstr)

    for img in glob.glob(fp_img):
        with rasterio.open(img) as src:
            height = src.height
            width = src.width
            src_transform = src.transform
            out_meta = src.meta
            mask_nodata = np.ones([height, width], dtype=np.uint8)
            for i in range(src.count):
                mask_nodata = mask_nodata & src.read_masks(i+1)
        out_meta.update({"count": 1, "dtype": 'uint8', 'nodata': 0})

        mask = rasterio.features.geometry_mask(bound_shp['geometry'], (height, width), src_transform, invert=True, all_touched=True).astype('uint8')
        mask = mask & mask_nodata
        mask = mask*255
        # label = img.replace('.tif', '_mask.tif')
        label_dir = os.path.dirname(fp_img) + '_mask'
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        label = os.path.join(label_dir, os.path.basename(img))
        with rasterio.open(label, 'w', compress='RAW', **out_meta) as ras:
            ras.write(mask[np.newaxis, :, :])

# dir_img = r"/home/skm/SKM_OLD/WORK/Cloud_Remove_Planet/Version_2_full/img"
# dir_label = r"/home/skm/SKM_OLD/WORK/Cloud_Remove_Planet/Version_2_full/label"

# list_fp = glob.glob(dir_img + '/*tif')
# for fp_img in list_fp:
#     fp_shp = os.path.join(dir_label, os.path.basename(fp_img).replace('.tif', '.shp'))
#     create_mask(fp_img, fp_shp)


# dir_label = r"/home/skm/SKM_OLD/public/DA/8_GreenCover_GeoMineraba/GeoMin/data_mosaic_gdal/rs_oke"
# dir_img = r"/home/skm/SKM_OLD/public/DA/8_GreenCover_GeoMineraba/GeoMin/data_mosaic_gdal/tmp_rs_ver2"

dir_img = r"/home/skm/data_ml_mount/other_treecounting/bapcai/anh_set_crs"
dir_label = r"/home/skm/data_ml_mount/other_treecounting/bapcai/label_bufer"

# list_fp = glob.glob(dir_label + '/*shp')
list_fp = glob.glob(dir_img + '/*tif')
for fp_img in list_fp:
    name = os.path.basename(fp_img)
    print(name)
    name = name.replace('.tif','.shp')
    fp_shp = os.path.join(dir_label, name)
    create_mask(fp_img, fp_shp)