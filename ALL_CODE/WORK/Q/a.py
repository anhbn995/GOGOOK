import os, glob
import rasterio
import numpy as np
import geopandas as gpd
import rasterio.features
from tqdm.notebook import tqdm

def create_mask(img_path, shp_path, out_path_mask, value_mask=None):
    with rasterio.open(img_path) as src:
        meta = src.meta
        height, width = src.height, src.width
        tr = src.transform
        crs_img = src.crs
    df = gpd.read_file(shp_path)

    # check epsg
    if df.crs.to_string() != crs_img.to_string():
        df = df.to_crs(epsg=str(crs_img.to_epsg()))
    shapes = df['geometry']
    mask = rasterio.features.rasterize(shapes, out_shape=(height, width), transform=tr)
    if value_mask:
        mask = mask*value_mask

    meta.update({'count': 1, 'nodata': 0})
    with rasterio.open(out_path_mask, 'w', **meta) as dst:
        dst.write(np.array([mask]))   

def main_build_mask(input_img, input_shp, value_mask = 255):
    parent = os.path.join(os.path.dirname(input_img), 'mask')
    output_mask = os.path.join(parent, os.path.basename(input_img))
    if os.path.isfile(input_img) and os.path.isfile(input_shp) and output_mask[:-4] == '.tif':
        print('Waiting ... !')
        outdir = os.path.dirname(output_mask)
        os.makedirs(outdir, exist_ok=True)
        create_mask(input_img, input_shp, output_mask, value_mask)

    elif os.path.isdir(input_img) and os.path.isdir(input_shp) and output_mask[:-4] != '.tif':
        os.makedirs(output_mask, exist_ok=True)
        list_fp_shp = glob.glob(os.path.join(input_shp, '*.shp'))
        print(f'Co {len(list_fp_shp)} file duoc ve!')
        for fp_shp in tqdm(list_fp_shp, desc = 'All'):
            name_f = os.path.basename(fp_shp)
            print(name_f)
            fp_img = os.path.join(input_img, name_f.replace('.shp', '.tif'))
            out_fp_mask = os.path.join(output_mask, name_f.replace('.shp', '.tif'))
            create_mask(fp_img, fp_shp, out_fp_mask, value_mask)
    else:
        print('Dau vao khong hop le')