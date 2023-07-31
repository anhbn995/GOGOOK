import os
import uuid
import json
from osgeo import gdal, ogr, osr
import numpy as np

from osgeo.gdal import GA_ReadOnly

from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles


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


def _translate(src_path, dst_path, profile="deflate", profile_options={}, **options):
    output_profile = cog_profiles.get(profile)
    output_profile.update(dict(BIGTIFF="IF_SAFER"))
    output_profile.update(profile_options)
    config = dict(
        GDAL_NUM_THREADS="ALL_CPUS",
        GDAL_TIFF_INTERNAL_MASK=True,
        GDAL_TIFF_OVR_BLOCKSIZE="128",
    )
    cog_translate(
        src_path,
        dst_path,
        output_profile,
        config=config,
        quiet=True,
        **options,
    )
    return True


def clip_image_with_aois(aois, image_id, out_ids, out_dir, temp_dir, on_success=None, on_processing=None):
    image_path = '{}/{}.tif'.format(out_dir, uuid.UUID(image_id).hex)
    if on_processing:
        on_processing(0)
    result_data = []
    for index in range(len(aois)):
        clipped_image_data = clip_image(aois[index], image_path, out_ids[index], out_dir, temp_dir)
        meta_path = clipped_image_data['path'].replace('.tif', '.json')
        stretch(clipped_image_data['path'], meta_path, mode=1)
        result_data.append(clipped_image_data)
    return result_data

def clip_image(aoi, image_path, out_id, out_dir, temp_dir, extra=''):
    temp_dir = os.path.abspath(temp_dir)
    aoi_path = '{}/{}.geojson'.format(temp_dir, uuid.UUID(out_id).hex)
    out_path = '{}/{}{}.tif'.format(out_dir, uuid.UUID(out_id).hex, extra)
    pretranslate_out_path =  '{}/pre_translate_{}.tif'.format(temp_dir, uuid.UUID(out_id).hex)
    with open(aoi_path, "w") as editor:
        editor.write(json.dumps(aoi))

    # open raster and get its georeferencing information
    dsr = gdal.Open(image_path, gdal.GA_ReadOnly)
    gt = dsr.GetGeoTransform()
    srr = osr.SpatialReference()
    srr.ImportFromWkt(dsr.GetProjection())

    # open vector data and get its spatial ref
    dsv = ogr.Open(aoi_path)
    lyr = dsv.GetLayer(0)
    srv = lyr.GetSpatialRef()

    # make object that can transorm coordinates
    ctrans = osr.CoordinateTransformation(srv, srr)

    ds = gdal.OpenEx(aoi_path)
    layer = ds.GetLayer()
    feature = layer.GetFeature(0)
    # read the geometry and transform it into the raster's SRS
    geom = feature.GetGeometryRef()
    geom.Transform(ctrans)
    # get bounding box for the transformed feature
    minx, maxx, miny, maxy = geom.GetEnvelope()

    # compute the pixel-aligned bounding box (larger than the feature's bbox)
    left = minx - (minx - gt[0]) % gt[1]
    right = maxx + (gt[1] - ((maxx - gt[0]) % gt[1]))
    bottom = miny + (gt[5] - ((miny - gt[3]) % gt[5]))
    top = maxy - (maxy - gt[3]) % gt[5]

    gdal.Warp(pretranslate_out_path, image_path, cutlineDSName=aoi_path, outputBounds= [left, right, bottom, top],xRes= abs(gt[1]), yRes= abs(gt[5]), cropToCutline=True)
    _translate(pretranslate_out_path, out_path)

    return {
        'geom': aoi,
        'id': out_id,
        'path': out_path,
        'bbox': [miny, minx, maxy, maxx]
    }

def stretch_image(input, output, mode=1):
    ds = gdal.Open(input, gdal.GA_ReadOnly)
    bcount = ds.RasterCount
    rows = ds.RasterXSize
    cols = ds.RasterYSize

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(output, rows, cols, bcount, gdal.GDT_Byte)
    outdata.SetGeoTransform(ds.GetGeoTransform())  ##sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())  ##sets same projection as input
    for i in range(bcount):
        band = np.array(ds.GetRasterBand(i + 1).ReadAsArray())
        nodatamask = (band == np.nan)
        band1 = band.astype(float)
        band1[nodatamask] = np.nan
        if mode == 1:  # cumulative
            p2 = np.nanpercentile(band1, 1)
            p98 = np.nanpercentile(band1, 99)
        else:  # standard deviation
            mean = np.nanmean(band1)
            std = np.nanstd(band1)
            p2 = mean - std * 2
            p98 = mean + std * 2
        band1 = None
        band = np.interp(band, (p2, p98), (1, 255)).astype(int)
        band[nodatamask] = 0
        outdata.GetRasterBand(i + 1).WriteArray(band)
        outdata.GetRasterBand(i + 1).SetNoDataValue(0)
        band = None
    outdata.FlushCache()  ##saves to disk!!
    outdata = None
    # close dataset
    ds = None


def stretch_without_json(input, mode=1, min=2, max=98):
    mode = int(mode)
    min = int(min)
    max = int(max)
    from osgeo import gdal_array
    raster = gdal_array.LoadFile(input)
    if len(np.shape(raster)) == 2:
        raster = [raster]
    bcount, _, _ = np.shape(raster)
    strecth_arr = []
    for i in range(bcount):
        band = raster[i]
        nodatamask = (band == 0)
        band1 = band.astype(float)
        band1[nodatamask] = np.nan
        nodatamask = (band == -9999)
        band1[nodatamask] = np.nan
        if mode == 1:  # cumulative
            p2 = (np.nanpercentile(band1, min))
            p98 = (np.nanpercentile(band1, max))
        else:  # standard deviation
            mean = np.nanmean(band1)
            std = np.nanstd(band1)
            p2 = mean - std * 2
            p98 = mean + std * 2
        strecth_arr.append({'p2': p2, 'p98': p98})
    return strecth_arr


def stretch(input, output, mode=1, min=2, max=98):
    strecth_arr = stretch_without_json(input=input, mode=mode, min=min, max=max)
    data = {
        'stretches': strecth_arr
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


def reproject_image_goc(src_path, dst_path, dst_crs='EPSG:4326'):
    temp_path = dst_path.replace('.tif', 'temp.tif')
    option = gdal.TranslateOptions(gdal.ParseCommandLine("-co \"TFW=YES\""))
    gdal.Translate(temp_path, src_path, options=option)
    option = gdal.WarpOptions(gdal.ParseCommandLine("-t_srs {}".format(dst_crs)))
    gdal.Warp(dst_path, temp_path, options=option)
    os.remove(temp_path)
    return True


def reproject_image(src_path, dst_path, dst_crs='EPSG:4326'):
    import rasterio
    with rasterio.open(src_path) as ds:
        nodata = ds.nodata or 0
    if ds.crs.to_string() != dst_crs:
        print(f'convert to {dst_crs}')
        temp_path = dst_path.replace('.tif', 'temp.tif')
        option = gdal.TranslateOptions(gdal.ParseCommandLine("-co \"TFW=YES\""))
        gdal.Translate(temp_path, src_path, options=option)
        option = gdal.WarpOptions(gdal.ParseCommandLine("-t_srs {} -dstnodata {}".format(dst_crs, nodata)))
        gdal.Warp(dst_path, temp_path, options=option)
        os.remove(temp_path)
    else:
        import shutil
        print(f'coppy image to {dst_path}')
        shutil.copyfile(src_path, dst_path)
        print('done coppy')
    return True



def compute_bound(img_path):
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


def rgb_2_hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)