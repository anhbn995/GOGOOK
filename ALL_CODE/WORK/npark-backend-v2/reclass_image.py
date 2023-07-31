from skimage.morphology import remove_small_objects
import rasterio
import numpy as np


def calc_forest(input_path, out_path, x_size, y_size, classification_image):
    meta = {}
    with rasterio.open(input_path) as r:
        a = r.read()
        out_meta = r.meta

    with rasterio.open(classification_image) as ds_classification:
        green_cover = np.count_nonzero(ds_classification.read(1) == 1) * x_size * y_size / 1000000

    mask = remove_small_objects(a >= 0.4, min_size=50)
    g = a * mask
    with rasterio.open(out_path, "w", **out_meta, compress='lzw') as dest:
        dest.write(g)
    with rasterio.open(out_path) as ds_forest:
        mask_forest = ds_forest.read(1)
        meta['FOREST_COVER_AREA'] = np.count_nonzero(mask_forest) * x_size * y_size / 1000000
        meta['NON_FOREST_COVER_AREA'] = green_cover - meta['FOREST_COVER_AREA']
    return meta


def calc_forest_segment(input_path, out_path, x_size, y_size, classification_image):
    meta = {}
    with rasterio.open(input_path) as r:
        a = r.read()
        out_meta = r.meta

    with rasterio.open(classification_image) as ds_classification:
        green_cover = np.count_nonzero(ds_classification.read(1) == 1) * x_size * y_size / 1000000

    mask = remove_small_objects(a >= 0.4, min_size=50)
    g = a * mask
    with rasterio.open(out_path, "w", **out_meta, compress='lzw') as dest:
        dest.write(g)
    with rasterio.open(out_path) as ds_forest:
        meta['FOREST_COVER_AREA'] = np.count_nonzero(ds_forest.read(1)) * x_size * y_size / 1000000
        meta['NON_FOREST_COVER_AREA'] = green_cover - meta['FOREST_COVER_AREA']
    return meta
