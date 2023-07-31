import json

import fiona
from flask import request

from app.job_queue import job_wrapper, QueueType
from app.job_queue.image_processors.crop_scout_analytic_processor import CropScoutAnalyticProcessor
from config.default import ROOT_DATA_FOLDER
from app.utils.response import success

@job_wrapper(QueueType.UTILITY)
def crop_scout_analyze():
    payload_mapping = {
        'image_path': 'image_path',
        'image_id': 'image_id',
        'result_id': 'result_id',
        'index_red': 'index_red',
        'index_green': 'index_green',
        'index_blue': 'index_blue',
        'index_nir': 'index_nir',
        'selected_values': 'selected_values',
        'meta': 'meta',
        "vector_id": "vector_id"
    }
    return CropScoutAnalyticProcessor, request.data, payload_mapping


def result_bbox():
    payload = json.loads(request.data)
    path = payload.get('path')
    file_path = f'{ROOT_DATA_FOLDER}/{path}'
    geom_ds = fiona.open(file_path)
    return success(list(geom_ds.bounds))
