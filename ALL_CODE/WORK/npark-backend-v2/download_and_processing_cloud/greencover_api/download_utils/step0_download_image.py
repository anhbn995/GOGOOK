import os
import sys
import glob
import json
import shutil
import requests

from download_and_processing_cloud.greencover_api.download_utils.utils import write_json_file
from download_and_processing_cloud.greencover_api.download_utils.from_url import main as download_image_from_url


def get_weights(weight_path):
    all_weights = {}
    list_weights = glob.glob(os.path.join(weight_path, '*.h5'))
    for i in list_weights:
        if 'cloud' in os.path.basename(i):
            all_weights.update({'cloud': i})
        elif 'green' in os.path.basename(i):
            all_weights.update({'green': i})
        elif 'water' in os.path.basename(i):
            all_weights.update({'water': i})
        else:
            raise Exception("Name of weights contains name of class.")
    if len(all_weights.keys()) != 3:
        raise Exception("Not enough file weight, please check %s" %
                        (weight_path))
    return all_weights


def get_token(login_url, user_info):
    login_info = requests.post(login_url, json=user_info)
    if login_info.ok:
        infomation = login_info.json()
        token = 'Bearer ' + infomation['body']['token']
    else:
        raise Exception("Can't get token, please check user id")
    return token


def main(folder_path, temp_path, name_ws, name_aoi, month, year, start_date, end_date, CLOUD_COVER):
    login_url = "https://auth.eofactory.ai/login"
    input_url = 'https://api-aws.eofactory.ai/api/workspaces?region=sea'
    user_info = {"email": 'quyet.nn@eofactory.ai',
                 "password": 'Quyet135667799'}
    token = get_token(login_url, user_info)
    folder_download = os.path.join(
        workspace_path, 'download_list_%s_%s.json' % (str(month), str(year)))
    name_json_file = folder_download
    try:
        f = open(name_json_file)
        data = json.load(f)
    except:
        data = {'workspace_path': "",
                'temp_path': temp_path,
                'list_image': {},
                'weights': {},
                'AOI': []}
        write_json_file(data, name_json_file)

    workspace_path, list_month_folder = download_image_from_url(folder_path, name_ws, name_aoi, input_url, token, month, year,
                                                                start_date, end_date, CLOUD_COVER, data, name_json_file)
    print("Finished all")
    return workspace_path, list_month_folder


if __name__ == "__main__":
    month = [6, 7, 8, 9]
    year = [2021]
    start_date = 1
    # end_date = None is download all
    end_date = None
    CLOUD_COVER = 1.0

    # Check before run
    name_aoi = "new_aoi"
    name_ws = "Green Cover Npark Singapore"
    temp_path = '/home/geoai/geoai_data_test2'
    folder_path = '/home/nghipham/Desktop/Jupyter/data/DucAnh/WORK/BoNongNghiep_Indo/IMG'
    workspace_path = main(folder_path, temp_path, name_ws,
                          name_aoi, month, year, start_date, end_date, CLOUD_COVER)
