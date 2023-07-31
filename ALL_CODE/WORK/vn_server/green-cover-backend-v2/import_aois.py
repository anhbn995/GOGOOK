import rasterio
import requests
from config.default import HOSTED_ENDPOINT
import geojson
from pathlib import Path
import geopandas as gpd
import os
import ast
import gdal
import numpy as np
from rasterio.warp import reproject, Resampling

def store_aois(payload):
    url = f'{HOSTED_ENDPOINT}/internal/aois'

    print(url)
    r = requests.post(url, json=payload)
    if not r.ok:
        print("status_code: ", r.status_code)
        raise Exception('Fail to upload result')


def align_images(input, reference, output):
    # Source
    src_filename = input
    src = gdal.Open(src_filename, gdal.gdalconst.GA_ReadOnly)
    dtype = src.GetRasterBand(1).DataType

    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()

    # We want a section of source that matches this:
    match_filename = reference
    match_ds = gdal.Open(match_filename, gdal.gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize

    # Output / destination
    dst_filename = output
    dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, src.RasterCount, dtype)
    dst.SetGeoTransform(match_geotrans)
    dst.SetProjection(match_proj)

    # Do the work
    gdal.ReprojectImage(src, dst, src_proj, match_proj, gdal.gdalconst.GRA_Bilinear)

    del dst  # Flush
    src = None
    match_ds = None
    dst = None

def convert_profile(src_path, dst_path, out_path):
    kwargs = None
    _info = gdal.Info(dst_path, format='json')
    xmin, ymin = _info['cornerCoordinates']['lowerLeft']
    xmax, ymax = _info['cornerCoordinates']['upperRight']

    with rasterio.open(dst_path) as dst:
        dst_transform = dst.transform
        kwargs = dst.meta
        kwargs['transform'] = dst_transform
        dst_crs = dst.crs

    with rasterio.open(src_path) as src:
        window = window_from_extent(xmin, xmax, ymin, ymax, src.transform)
        src_transform = src.window_transform(window)
        data = src.read()

        with rasterio.open(out_path, 'w', **kwargs) as dst:
            for i, band in enumerate(data, 1):
                _band = src.read(i, window=window)
                dest = np.zeros_like(_band)
                reproject(
                    _band,
                    dest,
                    src_transform=src_transform,
                    src_crs=src.crs,
                    dst_transform=src_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

                dst.write(dest, indexes=i)

def window_from_extent(xmin, xmax, ymin, ymax, aff):
    col_start, row_start = ~aff * (xmin, ymax)
    col_stop, row_stop = ~aff * (xmax, ymin)
    return ((int(row_start), int(row_stop)), (int(col_start), int(col_stop)))

if __name__ == '__main__':

    # t9 = '/home/geoai/geoai_data_test/8_GreenCover_GeoMineraba/Thang/Locus Provinsi Kalimantan Selatan/T9/classification/20210913_014859_81_2456_3B_AnalyticMS_clip_mosaic_19092021_color_color_final.tif'
    # t10 = '/home/geoai/geoai_data_test/8_GreenCover_GeoMineraba/Thang/Locus Provinsi Kalimantan Selatan/T10/classification/20211027_014914_24_2423_3B_AnalyticMS_clip_color_color_final.tif'
    # result = '/home/thang/Downloads/result.tif'
    # convert_profile(t9, t10,result)
    # import pdb
    # pdb.set_trace()
    # list_geojson = [
    #     {
    #         'path': '/home/thang/Desktop/2_aoi_singapore.geojson',
    #         'name': 'AOI SELECTED'
    #
    #     },
    #     {
    #         'path': '/home/thang/Downloads/new_aoi.geojson',
    #         'name': 'AOI Entire Singapore',
    #
    #     }
    # ]
    list_geojson = []
    folder_vector = '/home/thang/Desktop/projects/npark/vectors'
    list_kml = []
    payload = []
    gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
    # for path in Path(folder_vector).rglob('*'):
    # kml_path = '/home/thang/Desktop/SanFrancisco/Sentinel2/AOI/SFO.geojson'
    # try:
    #     df = gpd.read_file(kml_path)
    #     df = df.explode()
    #     print(os.stat(kml_path).st_size)
    #     payload.append({
    #         'name': 'SanFrancisco',
    #         'geometry': ast.literal_eval(df.to_json().replace("null", '""'))
    #     })
    
    kml_path = '/home/skm/Downloads/greenintheWorld/SanJose.geojson'
    # print(os.stat(kml_path).st_size)
    try:
        df = gpd.read_file(kml_path)
        df = df.explode()
        print(os.stat(kml_path).st_size)
        payload.append({
            'name': 'SanJose',
            'geometry': ast.literal_eval(df.to_json().replace("null", '""'))
        })


    except Exception as e:
        print(e)

    # for item in list_geojson:
    #     with open(item['path']) as f:
    #         gj = geojson.load(f)
    #     payload.append({
    #         'name': item['name'],
    #         'geometry': gj
    #     })
    # print(payload)
    store_aois(payload)
