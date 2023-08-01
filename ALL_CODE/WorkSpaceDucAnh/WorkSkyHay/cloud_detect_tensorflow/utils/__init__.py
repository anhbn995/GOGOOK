import os
import glob
from download_and_processing_cloud.greencover_api.cloud_remove.export_cloud_to_nodata import sort_list_file_by_cloud
from download_and_processing_cloud.greencover_api.cloud_remove.step2_cloud_remove import main as remove_cloud

def get_list_fp(folder_dir, type_file = '*.tif'):
        list_fp = []
        for file_ in glob.glob(os.path.join(folder_dir, type_file)):
            head, tail = os.path.split(file_)
            list_fp.append(os.path.join(head, tail))
        return list_fp

def cloud_remove_month(input_path, tmp_path, weight_path_cloud, base_image_gdal, name):
    if os.path.exists(os.path.join(tmp_path, name)):
        num_img = len(glob.glob(os.path.join(input_path,'*.tif')))
        num_predict = len(glob.glob(os.path.join(tmp_path, name, 'data_genorator_01/predict_float/cloud','*.tif')))
        if num_img == num_predict:
            print("Exist cloud remove result %s"%(name))
            dir_predict_cloud_tmp = os.path.join(tmp_path, name, 'data_genorator_01/predict_float')
            out_cloud = os.path.join(dir_predict_cloud_tmp, "cloud")
            list_fn_sort = sort_list_file_by_cloud(dir_predict_cloud_tmp)

        else:
            print("Cloud remove result folder %s is empty"%(name))
            print("Run cloud remove...")
            list_fp_img_selected = get_list_fp(input_path)
            sort_amount_of_clouds = True
            first_image = None
            tmp_dir = os.path.join(tmp_path, name)
            out_cloud, list_fn_sort = remove_cloud(list_fp_img_selected, tmp_dir, weight_path_cloud, sort_amount_of_clouds, first_image)
    else:
        print("Cloud remove result folder %s is empty"%(name))
        print("Run cloud remove...")
        list_fp_img_selected = get_list_fp(input_path)
        sort_amount_of_clouds = True
        first_image = None
        tmp_dir = os.path.join(tmp_path, name)
        out_cloud, list_fn_sort = remove_cloud(list_fp_img_selected, tmp_dir, weight_path_cloud, sort_amount_of_clouds, first_image)
    return out_cloud, list_fn_sort