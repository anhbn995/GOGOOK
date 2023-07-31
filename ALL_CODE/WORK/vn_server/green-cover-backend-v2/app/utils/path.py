import uuid
import os
from config.default import ROOT_DATA_FOLDER


def get_temp_path():
    return './temp'

def make_temp_folder():
    temp_folder = uuid.uuid4().hex
    cache_dir = get_temp_path()
    path = '{}/{}'.format(cache_dir, temp_folder)
    os.makedirs(path)
    return path

def path_2_abs(path):
    return ROOT_DATA_FOLDER + path


def make_temp_folder_in_root_data_folder():
    temp_folder = uuid.uuid4().hex
    path = '{}/data/temp/{}'.format(ROOT_DATA_FOLDER, temp_folder)
    os.makedirs(path)
    return path