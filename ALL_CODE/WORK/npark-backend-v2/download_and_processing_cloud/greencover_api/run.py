import os
import datetime
from config.default import NPARK_DATA_FOLDER
from download_and_processing_cloud.greencover_api.main import main as run_main
# from download_utils import download_data
from download_and_processing_cloud.greencover_api.download_utils.test_filter_cloud import main as download_data
# test


def check_exists_foler_with_name(folder_path, name):
    out_folder = os.path.join(folder_path, name)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    return out_folder


def get_current_time():
    now = datetime.datetime.now()
    month = now.month
    year = now.year
    list_year = [int(year)]
    list_month = [int(month)-1]
    return list_month, list_year


def main_download_and_predict(list_month, list_year, start_date, end_date, CLOUD_COVER, num_band=4, use_box=True, crs='EPSG:4326'):
    print("Download images at current time...")
    name_aoi = "new_aoi"
    name_ws = "Green Cover Npark Singapore"
    root_path = NPARK_DATA_FOLDER or os.getcwd()
    folder_path = os.path.join(root_path, 'data_projects')
    # đường dẫn thư mục trên con máy test có nhiệm vụ download ảnh (//192.168.4.104/eofactorytest)
    temp_path = '/home/geoai/geoai_data_test2'
    weight_path_cloud = os.path.join(root_path, 'weights', 'cloud_weights.h5')
    weight_path_green = os.path.join(root_path, 'weights', 'green_weights.h5')
    weight_path_water = os.path.join(root_path, 'weights', 'water_weights.h5')
    weight_path_forest = os.path.join(
        root_path, 'weights', 'forest_weights_v2.h5')
    # workspace_path, list_month_folder = download_data(folder_path, temp_path, name_ws, name_aoi,
    #                                                   list_month, list_year, start_date, end_date, CLOUD_COVER)
    workspace_path = "/home/skm/SKM16/Tmp/npark-backend-v2/data_projects/Green Cover Npark Singapore"
    print("Download: done")

    run_main(workspace_path, weight_path_cloud, weight_path_green, weight_path_water,
             weight_path_forest, num_band=num_band, use_box=use_box, crs=crs)


if __name__ == "__main__":
    start_date = 1
    end_date = None
    list_month, list_year = get_current_time()
    main_download_and_predict(list_month, list_year, start_date, end_date,
                              CLOUD_COVER=1.0, num_band=4, use_box=True, crs='EPSG:4326')
