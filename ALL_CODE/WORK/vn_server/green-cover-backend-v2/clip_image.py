import os
import shutil

import gdal, osr, ogr
import json
import geojson
from inject_image import _translate






def clip(image_path, aoi, out_path, temp_dir):
    aoi_path = '{}/aoi.geojson'.format(temp_dir)

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
    pre_translate = f'{temp_dir}/pre_translate.tif'
    gdal.Warp(pre_translate, image_path, cutlineDSName=aoi_path, outputBounds=[left, right, bottom, top], xRes=abs(gt[1]),
              yRes=abs(gt[5]), cropToCutline=True)
    _translate(pre_translate,out_path)

    os.remove(pre_translate)


if __name__ == '__main__':
    list_geojson = [
        {
            'path': '/home/thang/Desktop/2_aoi_singapore.geojson',
            'name': 'AOI SELECTED'

        },
        {
            'path': '/home/thang/Desktop/sing.geojson',
            'name': 'AOI Entire Singapore',

        }
    ]
    payload = []
    for item in list_geojson:
        with open(item['path']) as f:
            gj = geojson.load(f)
        payload.append({
            'name': item['name'],
            'geometry': gj
        })

    # store_aois(payload)
