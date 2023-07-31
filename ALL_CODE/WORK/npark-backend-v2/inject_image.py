import os
from osgeo import gdal
import json
import sys
import uuid
import argparse
import shutil
import numpy as np

from osgeo import osr, ogr
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

from app.utils.string import get_unique_id
from config.default import ROOT_DATA_FOLDER, SQLALCHEMY_DATABASE_URI


GDAL_DATATYPE_ARR = [
    'unknown',
    'uint8',
    'uint16',
    'int16',
    'uint32',
    'int32',
    'float32',
    'float64',
    'cint16',
    'cint32',
    'cfloat32',
    'cfloat64'
]


def get_file_size(path):
    return os.path.getsize(path)

def _translate(src_path, dst_path, profile="deflate", profile_options={}, **options):
    output_profile = cog_profiles.get(profile)
    output_profile.update(dict(BIGTIFF="IF_SAFER"))
    output_profile.update(profile_options)
    config = dict(GDAL_NUM_THREADS="ALL_CPUS",GDAL_TIFF_INTERNAL_MASK=True, GDAL_TIFF_OVR_BLOCKSIZE="128",)
    cog_translate(src_path,dst_path,output_profile,config=config,quiet=True,**options,)
    return True

def stretch_v2(path, output, min=2, max=98, nodata=0):
    from rio_tiler.io import COGReader
    abs_path = path
    with COGReader(abs_path) as cog:
        res = []
        try:
            stats = cog.stats(pmax=max, pmin=min, nodata=nodata)
            for _, value in stats.items():
                res.append({
                    'p2': value['pc'][0],
                    'p98': value['pc'][1],
                })
        except Exception:
            img = cog.preview()
    data = {
        'stretches': res
    }
    with open(output, 'w') as outfile:
        json.dump(data, outfile)

def calculate_metadata(file_path):
    raster = gdal.Open(file_path)
    bands_count = raster.RasterCount
    meta = raster.GetMetadata()
    gt = raster.GetGeoTransform()
    meta["PROJECTION"] = raster.GetProjection()
    meta["X_SIZE"] = raster.RasterXSize
    meta["Y_SIZE"] = raster.RasterYSize
    bands = []
    for i in range(bands_count):
        band = raster.GetRasterBand(i + 1)
        gdal.GetDataTypeName(band.DataType)
        if band.GetMinimum() is None or band.GetMaximum() is None:
            band.ComputeStatistics(0)
        band_meta = band.GetMetadata()
        band_meta["ID"] = i
        band_meta["NO_DATA_VALUE"] = band.GetNoDataValue()
        band_meta["MIN"] = band.GetMinimum()
        max_band_meta = band.GetMaximum()
        if band.GetMaximum() == np.inf:
            max_band_meta = "Infinity"
        band_meta["MAX"] = max_band_meta
        band_meta["DATA_TYPE"] = GDAL_DATATYPE_ARR[band.DataType]
        bands.append(band_meta)
    meta["BANDS"] = bands
    meta["MAX_ZOOM"] = get_optimal_zoom_level(raster)
    if meta["MAX_ZOOM"] < 10:
        meta["MAX_ZOOM"] = 14
    meta["MIN_ZOOM"] = 8
    meta["PIXEL_SIZE_X"] = gt[1]
    meta["PIXEL_SIZE_Y"] = -gt[5]
    import geopy.distance

    coords_origin = (gt[3], gt[0])
    coords_right = (gt[3], gt[0]+gt[1])
    coords_bottom = ( gt[3]+gt[5], gt[0])
    try:
        meta["PIXEL_SIZE_X_METER"] = geopy.distance.geodesic(coords_origin, coords_right).km * 1000
        meta["PIXEL_SIZE_Y_METER"] = geopy.distance.geodesic(coords_origin, coords_bottom).km * 1000
    except Exception as e:
        meta["PIXEL_SIZE_X_METER"] = meta["PIXEL_SIZE_X"]
        meta["PIXEL_SIZE_Y_METER"] = meta["PIXEL_SIZE_Y"]
    meta["ORIGIN_X"] = gt[0]
    meta["ORIGIN_Y"] = gt[3]
    return meta

def get_optimal_zoom_level(geo_tiff):
    import math
    geo_transform = geo_tiff.GetGeoTransform()
    degrees_per_pixel = geo_transform[1]
    radius = 6378137
    equator = 2 * math.pi * radius
    meters_per_degree = equator / 360
    resolution = degrees_per_pixel * meters_per_degree
    pixels_per_tile = 256
    zoom_level = math.log((equator/pixels_per_tile)/resolution, 2)
    MAX_ZOOM_LEVEL = 21
    optimal_zoom_level = min(round(zoom_level), MAX_ZOOM_LEVEL)
    return optimal_zoom_level

def compute_bound(img_path):
    from osgeo import gdal
    from osgeo.gdal import GA_ReadOnly

    data = gdal.Open(img_path, GA_ReadOnly)
    geoTransform = data.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * data.RasterXSize
    miny = maxy + geoTransform[5] * data.RasterYSize

    src_projection = osr.SpatialReference(wkt=data.GetProjection())
    tar_projection = osr.SpatialReference()
    tar_projection.ImportFromEPSG(4326)
    wgs84_trasformation = osr.CoordinateTransformation(src_projection, tar_projection)

    point_list = [[minx, miny],[minx, maxy],[maxx, maxy],[maxx, miny],[minx, miny]]
    tar_point_list = []

    for _point in point_list:
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(_point[0], _point[1])
        point.Transform(wgs84_trasformation)
        tar_point_list.append([point.GetX(), point.GetY()])

    geometry = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [tar_point_list]
                }
            }
        ]
    }
    data = None
    return geometry


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        '--input',
        help='Orginal Image Directory',
        required=True
    )
    args_parser.add_argument(
        '--name',
        help='Image name',
        required=True
    )
    args_parser.add_argument(
        '--user_id',
        help='User id',
        required=True
    )
    args_parser.add_argument(
        '--workspace',
        help='Workspace',
        required=True
    )
    args_parser.add_argument(
        '--cog',
        help='COG',
        required=True
    )
    param = args_parser.parse_args()
    input_path = param.input
    name = param.name
    workspace = param.workspace
    cog = param.cog
    user_id = param.user_id
    fid = uuid.uuid4()

    workspace_id = uuid.UUID(workspace)
    file_path = f'/data/{user_id}/{workspace_id.hex}/{fid.hex}.tif'
    output = f'{ROOT_DATA_FOLDER}{file_path}'
    output_stretch = f'{ROOT_DATA_FOLDER}/data/{user_id}/{workspace_id.hex}/{fid.hex}.json'
    tile_url = f'/tile/{user_id}/{workspace_id.hex}/{fid.hex}'+'/{z}/{x}/{y}'
    if not cog == 0:
        _translate(input_path, output)
    else:
        shutil.copyfile(input_path, output)

    stretch_v2(output, output_stretch)
    geometry = compute_bound(output)
    metadata = calculate_metadata(output)

    from sqlalchemy import create_engine
    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    with engine.connect() as connection:
        file_size = get_file_size(output)

        geometry_query = "ST_SetSRID(ST_GeomFromGeoJSON(\'{}\'),4326)".format(
            str(geometry.get('features')[0].get('geometry')).replace('\'', '\"'))
        del metadata["PROJECTION"]
        query = """
        INSERT INTO images(id, created_by, workspace_id, name, path, tile_url, meta, size, x_size, y_size, geometry, nodata, owner_id)
        VALUES (\'{}\', {}, \'{}\', \'{}\', \'{}\', \'{}\', \'{}\',{}, {}, {}, {}, 0, {})
        """.format(str(fid), user_id, str(workspace_id), name, file_path, tile_url, str(metadata).replace('\'', '\"'), file_size, metadata.get('X_SIZE', 0), metadata.get('Y_SIZE', 0), geometry_query, user_id)
        connection.execute(query)
    print(fid)
