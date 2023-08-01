import os
from task import celery
from lib.docker_executor import task_wrapper
from config.storage import ROOT_DATA_FOLDER
from lib.celery.indices_monitor.get_info_local import GetInfoLocal
from lib.indices_monitor.processors import sentinel2_local, planet_local
@celery.task(bind=True, name="tasks.indices_monitor.get_sentinel2_info_local", track_started=True, acks_late=True, base=GetInfoLocal)
@task_wrapper
def get_sentinel2_info_local(self, image_path, uuid, task_detail):
    image_path = f'{ROOT_DATA_FOLDER}{image_path}'
    info = {}
    for key in sentinel2_local.EXPRESSION:
        processor = sentinel2_local.Sentinel2Local(image_path, key)
        key_info = processor.get_info()
        info[key] = key_info
    self.return_value = {
        'info': info,
        'uuid': uuid
    }
    return self.return_value

@celery.task(bind=True, name="tasks.indices_monitor.get_planet_info_local", track_started=True, acks_late=True, base=GetInfoLocal)
@task_wrapper
def get_planet_info_local(self, image_path, uuid, task_detail):
    image_path = f'{ROOT_DATA_FOLDER}{image_path}'
    info = {}
    for key in planet_local.EXPRESSION:
        processor = planet_local.Planet8BandsLocal(image_path, key)
        key_info = processor.get_info()
        info[key] = key_info
    self.return_value = {
        'info': info,
        'uuid': uuid
    }
    return self.return_value

