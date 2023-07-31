import os
from green_cover.step_5_classification import run_segmentation, combine_all


def classification(list_removecloud_img, weight_path_green, weight_path_water, result_path):
    for j in list_removecloud_img:
        print("******Green cover %s******"%(os.path.basename(j).split('.')[0]))
        tmp_folder = os.path.join(os.path.dirname(result_path), 'tmp')
        if not os.path.exists(tmp_folder):
            os.mkdir(tmp_folder)
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        result_green_tmp, result_water_tmp = run_segmentation(j, weight_path_green, weight_path_water, tmp_folder)
        print("Done")
        print("Combine result...")
        combine_all(j, result_green_tmp, result_water_tmp, result_path, tmp_folder)
        print("Done")
        print("\n")