import math
import numpy as np
import geopy.distance

from osgeo import gdal, osr, ogr
from rio_tiler.io import COGReader


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


EARTH_RADIUS = 6378137
EARTH_EQUATOR = 2 * math.pi * EARTH_RADIUS


def _get_optimal_zoom_level(file_path):
    print(file_path, "___________")
    cog = COGReader(file_path)
    return cog.minzoom, cog.maxzoom


def _get_gdal_dtype_name(gdal_dtype):
    return GDAL_DATATYPE_ARR[gdal_dtype]


def _get_dataset_band_stats(dataset ,band_idx):
    band = dataset.GetRasterBand(band_idx + 1)
    if band.DataType == 1:
        band_stats = band.GetMetadata()
        band_stats["MAX"] = 255
        band_stats["ID"] = band_idx
        band_stats["NO_DATA_VALUE"] = 0
        band_stats["MIN"] = 1
        band_stats["DATA_TYPE"] = _get_gdal_dtype_name(band.DataType)

        return band_stats
    if band.GetMinimum() is None or band.GetMaximum() is None:
        band.ComputeStatistics(0)
    band_stats = band.GetMetadata()

    band_max_value = band.GetMaximum()
    if band.GetMaximum() == np.inf:
        band_max_value = "Infinity"

    band_stats["MAX"] = band_max_value
    band_stats["ID"] = band_idx
    band_stats["NO_DATA_VALUE"] = band.GetNoDataValue()
    band_stats["MIN"] = band.GetMinimum()
    band_stats["DATA_TYPE"] = _get_gdal_dtype_name(band.DataType)

    return band_stats


def _get_bbox_from_geotransform(geo_transform, projection, img_width, img_height):
    minx = geo_transform[0]
    maxy = geo_transform[3]
    maxx = minx + geo_transform[1] * img_width
    miny = maxy + geo_transform[5] * img_height

    src_projection = osr.SpatialReference(wkt=projection)
    tar_projection = osr.SpatialReference()
    tar_projection.ImportFromEPSG(4326)
    wgs84_trasformation = osr.CoordinateTransformation(src_projection, tar_projection)

    point_list = [[minx, miny],[minx, maxy],[maxx, maxy],[maxx, miny],[minx, miny]]
    tar_point_list = []

    for _point in point_list:
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint_2D(_point[0], _point[1])
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
    return geometry


def calculate_metadata(file_path, geometry=None):
    raster = gdal.Open(file_path)
    bands_count = raster.RasterCount
    geo_transform = raster.GetGeoTransform()
    projection = raster.GetProjection()

    bands = []
    for i in range(bands_count):
        band_stats = _get_dataset_band_stats(raster, i)
        bands.append(band_stats)
    
    meta = raster.GetMetadata()
    meta["X_SIZE"] = raster.RasterXSize
    meta["Y_SIZE"] = raster.RasterYSize
    meta["BANDS"] = bands
    meta["MIN_ZOOM"], meta["MAX_ZOOM"] = _get_optimal_zoom_level(file_path)
    if meta["MAX_ZOOM"] < 18:
        meta["MAX_ZOOM"] = 18
    meta["PIXEL_SIZE_X"] = geo_transform[1]
    meta["PIXEL_SIZE_Y"] = -geo_transform[5]

    coords_origin = (geo_transform[3], geo_transform[0])
    coords_right = (geo_transform[3], geo_transform[0]+geo_transform[1])
    coords_bottom = ( geo_transform[3]+geo_transform[5], geo_transform[0])
    try:
        meta["PIXEL_SIZE_X_METER"] = geopy.distance.geodesic(coords_origin, coords_right).km * 1000
        meta["PIXEL_SIZE_Y_METER"] = geopy.distance.geodesic(coords_origin, coords_bottom).km * 1000
    except Exception:
        meta["PIXEL_SIZE_X_METER"] = meta["PIXEL_SIZE_X"]
        meta["PIXEL_SIZE_Y_METER"] = meta["PIXEL_SIZE_Y"]
    meta["ORIGIN_X"] = geo_transform[0]
    meta["ORIGIN_Y"] = geo_transform[3]
    meta["geometry"] = _get_bbox_from_geotransform(geo_transform, projection, raster.RasterXSize, raster.RasterYSize)
    return meta


def get_bandcount(image_path):
    raster = gdal.Open(image_path)
    return raster.RasterCount


def get_quantile_schema(img):
    qt_scheme = []
    with COGReader(img) as cog:
        stats = cog.stats()
        for _, value in stats.items():
            qt_scheme.append({
                'p2': value['pc'][0],
                'p98': value['pc'][1],
            })
    return qt_scheme