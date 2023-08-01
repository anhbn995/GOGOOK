import os
from task import celery
from lib.docker_executor import task_wrapper
from config.storage import ROOT_DATA_FOLDER, TEMP_FOLDER
from lib.models.classification.green_cover.models.predict_green import main_green_normal
from lib.models.classification.green_cover.models.predict_water import main_water_normal
from lib.models.classification.green_cover.models.main_union_class import main_union_class
from lib.models.classification.green_cover.calculate_green_to_non_index.cal_object_area import main_cal_many_class_area
from lib.celery.ml.predict_task import PredictTask
from config.ml import GREEN_WEIGHTS, WATER_WEIGHTS, DICT_COLORMAP, WIEGHT_STORAGE
from lib.mosaic.utils import get_image_stats
import uuid

@celery.task(bind=True, name="tasks.ml.predict_green", track_started=True, acks_late=True, base=PredictTask)
@task_wrapper
def predict_green(self, image_path, task_detail):
    image_path = f'{ROOT_DATA_FOLDER}{image_path}'
    abso_path = f'/data/{task_detail.get("user_id")}/{task_detail.get("project_id")}/{self.request.id}.tif'
    stats = get_image_stats(image_path)
    self.return_value = {
        'path': abso_path,
        'stats': stats
    }
    output_path = f'{ROOT_DATA_FOLDER}{abso_path}'
    main_green_normal(image_path, output_path, GREEN_WEIGHTS)
    print("Predict green is done!!")

@celery.task(bind=True, name="tasks.ml.predict_water", track_started=True, acks_late=True, base=PredictTask)
@task_wrapper
def predict_water(self, image_path, task_detail):
    image_path = f'{ROOT_DATA_FOLDER}{image_path}'
    abso_path = f'/data/{task_detail.get("user_id")}/{task_detail.get("project_id")}/{self.request.id}.tif'
    stats = get_image_stats(image_path)
    self.return_value = {
        'path': abso_path,
        'stats': stats
    }
    output_path = f'{ROOT_DATA_FOLDER}{abso_path}'
    main_water_normal(image_path, output_path, WATER_WEIGHTS)
    print("Predict water is done!!")

@celery.task(bind=True, name="tasks.ml.predict_union", track_started=True, acks_late=True, base=PredictTask)
@task_wrapper
def predict_union(self, image_path, image_source, task_detail):
    image_path = f'{ROOT_DATA_FOLDER}{image_path}'
    abso_path_forest = f'/data/{task_detail.get("user_id")}/{task_detail.get("project_id")}/{str(uuid.uuid4())}.tif'
    abso_path_green = f'/data/{task_detail.get("user_id")}/{task_detail.get("project_id")}/{str(uuid.uuid4())}.tif'
    main_union_class(image_path, abso_path_green, abso_path_forest, WIEGHT_STORAGE, DICT_COLORMAP, TEMP_FOLDER)

    forest_obj_area = main_cal_many_class_area(abso_path_forest, list_value_class=[1,2,3,4,5], type_image=image_source)
    green_obj_area = main_cal_many_class_area(abso_path_green, list_value_class=[1,2,3,5], type_image=image_source)

    self.return_value = [
        {
        'type': 'green_cover',
        'path': abso_path_green,
        'stats': green_obj_area
        },
        {
        'type': 'forest_cover',
        'path': abso_path_forest,
        'stats': forest_obj_area
        }
    ]
    print("Predict union is done!!")
    return self.return_value