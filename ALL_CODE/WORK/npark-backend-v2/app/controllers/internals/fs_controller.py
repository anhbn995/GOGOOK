import json
import os
import shutil
import logging
import glob

from flask import request

from app.utils.response import success, not_found
from app.utils.path import path_2_abs
from app.utils import get_folder_size
from app.job_queue.fs import copy
from app.job_queue import utility_task_queue
import geopandas as gpd
from app.utils.path import make_temp_folder

logger = logging.getLogger('geoai')
from pathlib import Path


def mkdirs():
    payload = json.loads(request.data)
    data = []
    if payload.get('paths'):
        for path in payload.get('paths'):
            abs_path = path_2_abs(path)
            data.append(abs_path)
            if not os.path.exists(abs_path):
                os.makedirs(abs_path)
        return success(data)
    path = payload.get('path')
    abs_path = path_2_abs(path)
    data.append(abs_path)
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
    return success(data)


def delete_dir():
    payload = json.loads(request.data)
    path = payload.get('path')
    abs_path = path_2_abs(path)
    shutil.rmtree(abs_path, ignore_errors=True)
    return success('success')


def delete_dirs():
    payload = json.loads(request.data)
    paths = payload.get('paths')
    for path in paths:
        abs_path = path_2_abs(path)
        shutil.rmtree(abs_path, ignore_errors=True)
    return success('success')


def size_dir():
    payload = json.loads(request.data)
    path = payload.get('path')
    abs_path = path_2_abs(path)
    size = 0
    try:
        size = get_folder_size(abs_path) / 2 ** 20
    except Exception as e:
        not_found(message=str(e))
    return success({'size': size})


def copy_file():
    payload = json.loads(request.data)
    src_path = payload.get('src')
    dest_path = payload.get('dest')
    try:
        abs_src_path = path_2_abs(src_path)
        abs_dest_path = path_2_abs(dest_path)
        shutil.copyfile(abs_src_path, abs_dest_path)
    except Exception as e:
        not_found(message=str(e))
    return success('success')


def copy_files():
    payload = json.loads(request.data)
    task = utility_task_queue.enqueue(
        copy,
        payload.get('task_id'),
        payload.get('src'),
        payload.get('dest'),
        job_timeout='1h'
    )
    return success({
        'job_id': task.id,
        'queue_name': 'utility',
    })


def store_file():
    _file = request.files.get('file')
    store_path = request.form.get('path')
    abs_path = path_2_abs(store_path)
    _file.save(abs_path)
    _file.stream.close()
    return success('success')


def store_file_shp():
    _file = request.files.get('file')
    store_path = request.form.get('path')
    print(store_path)
    abs_path = path_2_abs(store_path)
    _file.save(abs_path)
    _file.stream.close()
    data_frame = gpd.read_file(abs_path, driver='GeoJSON')
    shp_path = os.path.splitext(abs_path)[0];
    data_frame.to_file(f"{shp_path}.shp", driver='ESRI Shapefile')
    return success('success')


def delete_file():
    payload = json.loads(request.data)
    paths = payload.get('paths')
    for path in paths:
        abs_path = path_2_abs(path)
        for file in glob.glob(abs_path):
            if os.path.exists(file):
                os.remove(file)

        # Delete pix file (raw image)
        pix = abs_path.split('/')
        pix.insert(-1, 'PIX')
        for pix_file in glob.glob('/'.join(pix)):
            if os.path.exists(pix_file):
                os.remove(pix_file)

        basename = os.path.splitext(os.path.basename(path))[0]
        dir_folder = os.path.dirname(path)
        for temp_path in Path(dir_folder).rglob(f'*{basename}*'):
            print(str(temp_path.resolve()))
            if os.path.exists(str(temp_path.resolve())):
                os.remove(str(temp_path.resolve()))

    return success('success')

if __name__ == '__main__':
    path ='/home/geoai/geoai_data_test2/data_npark_prod/data/2021/01/images/9f295e54e1ce432a81a4f16429b04202.tif'
    basename = os.path.splitext(os.path.basename(path))[0]
    dir_folder = os.path.dirname(path)
    for temp_path in Path(dir_folder).rglob(f'*{basename}*'):


        if os.path.exists(str(temp_path.resolve())):
            print(str(temp_path.resolve()))
            # os.remove(str(temp_path.resolve()))