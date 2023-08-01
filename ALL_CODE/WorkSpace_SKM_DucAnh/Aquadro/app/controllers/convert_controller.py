from flask import request
import json
from osgeo import ogr
from app.utils.response import success


def read_kml():
    result = []
    #path = request.json.get('file_path').replace('Z:\\tile-monitoring\\', '/home/geoai/geoai_data_test2/tile-monitoring/')
    path = request.json.get('file_path')
    driver = ogr.GetDriverByName('LIBKML')
    dataSource = driver.Open(path)
    layer = dataSource.GetLayer()

    for feat in layer:
        result.append({
            'properties': feat.items(),
            'geometry': json.loads(feat.geometry().ExportToJson())
        })

    return success(result)




