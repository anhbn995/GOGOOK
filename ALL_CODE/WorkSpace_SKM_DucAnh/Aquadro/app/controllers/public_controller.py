import os

import numpy as np
import rasterio.features

from flask import request
from flask import send_file

import shapely.wkt

from lib.searchers.creodias import CreodiasSearcher
from app.utils.indices_helper import *
from app.utils.response import success
from app.utils.db_service import get_geometry_from_field
from app.utils import calculate_area

from rio_tiler.io import COGReader
# from rio_tiler_pds.sentinel.aws import (
#     S2COGReader,  # COG
#     S1L1CReader
# )

from lib.image_reader.readers.aws.sentinel2 import S2COGReader
from app.services.image_readers import S1L1CReader
from lib.indices_monitor.processors import get_image_processor

from copy import deepcopy
from sklearn.cluster import KMeans

from shapely.geometry import shape


def search_images():
    payload = request.json
    source = payload.get('source')
    farm_boundary = shapely.wkt.loads(payload.get('geometry'))
    if 'sentinel' in source.lower():
        images = CreodiasSearcher(source, farm_boundary, payload.get('start_date'), payload.get('end_date')).search()
    else:
        # them cai loai anh khac
        images = []

    result = _transform_result(images, farm_boundary, source, payload.get('index'))
    return success(result)


def calculate_statistics():
    scene_ids = request.json.get('scene_ids')
    source = request.json.get('source')
    farm_boundary = shapely.wkt.loads(request.json.get('geometry'))
    index = request.json.get('index')
    images = [{'id': scene_id} for scene_id in scene_ids]
    return success(_transform_result(images, farm_boundary, source, index))


def _transform_result(images, farm_boundary, source, index):
    result = []
    for image in images:
        try:
            processor = get_image_processor(source)(image['id'], index, farm_boundary)
            image.update(processor.get_cutline_info())
            image['index'] = index
            result.append(deepcopy(image))
            del processor
        except Exception as e:
            print(e)
            continue
    return result


def download_image():
    payload = request.args
    scene_id = payload.get('scene_id')
    geometry = shapely.wkt.loads(get_geometry_from_field(payload.get('field')))
    index = payload.get('index')
    source = payload.get('source')

    processor = get_image_processor(source)(scene_id, index, geometry)
    file = processor.get_download_file()
    return send_file(file, mimetype='image/tiff', download_name=payload.get('file_name'))


def zoning():
    payload = request.args
    scene_id = payload.get('scene_id')
    field = payload.get('field')
    geometry = shapely.wkt.loads(get_geometry_from_field(field))
    zone = int(payload.get('zone'))

    index = payload.get('index')
    expression = index_expression(index)

    if index_source(index) == SENTINEL2:
        with S2COGReader(scene_id) as sen2:
            data = sen2.feature(expression=expression, shape=shapely.geometry.mapping(geometry), vrt_options={'CUTLINE_ALL_TOUCHED': True})
    if index_source(index) == SENTINEL1:
        os.environ['AWS_REQUEST_PAYER'] = 'requester'
        with S1L1CReader(scene_id) as sen1:
            data = sen1.feature(expression=expression, shape=shapely.geometry.mapping(geometry), vrt_options={'CUTLINE_ALL_TOUCHED': True})

    old_img_matrix = data.data[0]

    # Implement code của Bộ
    # Kmeans
    mask = data.mask / 255
    kmeans = KMeans(n_clusters=zone)
    kmeans.fit(np.array(old_img_matrix)[data.mask == 255].reshape(-1, 1))
    result = kmeans.predict(np.ma.array(old_img_matrix, mask=mask.astype('uint8').reshape(-1, 1)).reshape(-1, 1))
    labels = result.reshape(data.data.shape) + 1
    kmeans_matrix = (labels[0] * mask).astype('uint8')
    # End Kmeans

    # Start Lấp lỗ
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # img = cv2.morphologyEx(kmeans_matrix, cv2.MORPH_OPEN, kernel)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2)
    # kmeans_matrix = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
    # End

    shapes = {}
    for shp, val in rasterio.features.shapes(kmeans_matrix, transform=data.transform):
        if val > 0:
            val = int(val)
            if val not in shapes:
                shapes[val] = []
            shapes[val].append(shape(shp))

    res = []
    for i in shapes:
        geom = shapely.geometry.MultiPolygon(shapes[i]).intersection(geometry)
        res.append({
            'average': np.average(old_img_matrix[kmeans_matrix == i]),
            'geometry': shapely.geometry.mapping(geom),
            'area': calculate_area(geom)
        })
    from operator import itemgetter
    res = sorted(res, key=itemgetter('average'))

    return success(res)
