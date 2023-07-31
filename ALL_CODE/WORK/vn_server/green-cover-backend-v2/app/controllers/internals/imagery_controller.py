import json
import logging
import os.path
import uuid

import numpy as np
import rasterio
import requests
from app.utils.path import make_temp_folder
from flask import request
from rio_tiler.io import COGReader

from app.utils.response import success, error
from app.utils.path import path_2_abs
from app.utils.convert_l1c_to_l2a import convert_l1c_to_l2a
from app.job_queue import job_wrapper, QueueType
from app.job_queue.aws_image_downloader import AWSImageDownloader
from app.job_queue.driver_image_downloader import DriverImageDownloader
from app.job_queue.aws_raw_downloader import AWSRawDownloader

from config.default import ROOT_DATA_FOLDER
from config.default import PUBLIC_TILE_URL
from zipfile import ZipFile
from flask import request, send_from_directory
from app.utils.imagery import reproject_image
import shutil

logger = logging.getLogger('geoai')


@job_wrapper(QueueType.UTILITY)
def aws_download():
    payload = json.loads(request.data)

    if payload['image_type'] not in ['sentinel1_raw', 'sentinel2_raw', 'landsat8_raw']:
        payload_mapping = {
            'key': 'key',
            'image_type': 'image_type',
            'acquired': 'acquired',
            'extension': 'extension'
        }
        return AWSImageDownloader, request.data, payload_mapping
    else:
        payload_mapping = {
            'key': 'key',
            'image_type': 'image_type',
            'acquired': 'acquired',
        }
        return AWSRawDownloader, request.data, payload_mapping


@job_wrapper(QueueType.UTILITY)
def driver_download():
    payload_mapping = {
        'keys': 'keys'
    }
    return DriverImageDownloader, request.data, payload_mapping


def inspect():
    payload = json.loads(request.data)
    result = []
    for file_path in payload.get('file_paths'):
        with COGReader('{}{}'.format(ROOT_DATA_FOLDER, file_path)) as cog:
            result.append(list(cog.point(payload.get('longitude'), payload.get('latitude'))))
    return success(result)


def recalculate_quantile_scheme():
    payload = json.loads(request.data)
    abs_image_path = path_2_abs(payload.get('image_path'))
    url = f'{PUBLIC_TILE_URL}/public/images/quantile'
    payload['image_path'] = abs_image_path
    response = requests.post(url=url, json=payload)
    response = response.json()
    stretch_scheme_path = abs_image_path.replace('.tif', '.json')
    with open(stretch_scheme_path, 'w') as file_to_write:
        data = response.get('data')
        json.dump({'stretches': data}, file_to_write)
    return success("Success")


def stats():
    payload = json.loads(request.data)
    abs_image_path = path_2_abs(payload.get('file_path'))
    with COGReader(abs_image_path, nodata=np.nan) as cog:
        stat = cog.metadata(
            pmin=2.0,
            pmax=98.0,
            hist_options={'bins': 200}
        )
    return stat


def convert_product_type():
    payload = json.loads(request.data)
    return {
        'image_ids': convert_l1c_to_l2a(payload.get('image_ids'))
    }


def zip_image():
    payload = json.loads(request.data)
    image_paths = payload['images']
    dir_path = f"{ROOT_DATA_FOLDER}/data/zip"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    fid = uuid.uuid4()
    abs_file_path = f'{dir_path}/{fid.hex}.zip'
    with ZipFile(abs_file_path, 'w') as zip:
        check_same_name = {}
        for file in image_paths:
            if file['name'] in check_same_name:
                check_same_name[file['name']] += 1
            else:
                check_same_name[file['name']] = 0

            name = file["name"]
            if check_same_name[file['name']] > 0:
                name = f'{name}({check_same_name[file["name"]]})'

            zip.write(path_2_abs(file['path']), f'{name}.tif')

    return success({
        "path": abs_file_path
    })


def zip_vector():
    payload = json.loads(request.data)
    vector_paths = payload['vectors']
    dir_path = f"{ROOT_DATA_FOLDER}/data/zip"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    fid = uuid.uuid4()
    abs_file_path = f'{dir_path}/{fid.hex}.zip'
    with ZipFile(abs_file_path, 'w') as zip:
        check_same_name = {}
        for file in vector_paths:
            if file['name'] in check_same_name:
                check_same_name[file['name']] += 1
            else:
                check_same_name[file['name']] = 0

            name = file["name"]
            if check_same_name[file['name']] > 0:
                name = f'{name}({check_same_name[file["name"]]})'

            zip.write(path_2_abs(file['path']), f'{name}.geojson')

    return success({
        "path": abs_file_path
    })


def image_inspects():
    payload = json.loads(request.data)
    result = []
    month = []
    slope = None
    try:
        for file in payload.get('images'):
            with COGReader('{}{}'.format(ROOT_DATA_FOLDER, file['path'])) as cog:
                if not any(np.isnan(cog.point(payload.get('longitude'), payload.get('latitude')))):
                    if ('type' in file and file['type'] == 'plant_health'):
                        result.append(list(cog.point(payload.get('longitude'), payload.get('latitude')))[0])
                        month.append(file['month'])
                    else:
                        slope = list(cog.point(payload.get('longitude'), payload.get('latitude')))[0]
        return success({
            "labels": month,
            "values": result,
            "slope": slope
        })
    except Exception as e:
        return error(str(e))


def green_cover_change():
    payload = json.loads(request.data)
    try:
        image1 = f"{ROOT_DATA_FOLDER}{payload.get('images')[0]}"
        image2 = f"{ROOT_DATA_FOLDER}{payload.get('images')[1]}"
        with rasterio.open(image1) as ds1:
            band1 = ds1.read(1)
        with rasterio.open(image2) as ds2:
            band2 = ds2.read(1)
        x_size, y_size = get_pixel_size(image1)
        green_to_none = np.count_nonzero((band1 == 1) & (band2 != 1)) * x_size * y_size / 1000000
        none_to_green = np.count_nonzero((band1 != 1) & (band2 == 1)) * x_size * y_size / 1000000
        return success({
            "green_to_none": green_to_none,
            "none_to_green": none_to_green,
        })
    except Exception as e:
        raise e
        return error(str(e))


def get_pixel_size(image_path):
    temp_folder = make_temp_folder()
    temp_reproject = f'{temp_folder}/3857_mosaic.tif'
    reproject_image(image_path, temp_reproject, 'EPSG:3857')
    with rasterio.open(temp_reproject) as dt:
        x_size = dt.res[0]
        y_size = dt.res[1]
    shutil.rmtree(temp_folder)
    return x_size, y_size
