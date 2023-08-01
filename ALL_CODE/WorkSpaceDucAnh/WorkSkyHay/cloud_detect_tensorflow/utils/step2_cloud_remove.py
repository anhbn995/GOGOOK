import os
import glob
from tqdm import tqdm

from cloud_remove.export_cloud_to_nodata import main_cloud_to_nodata
from cloud_remove.normalize_data import main_gendata
from cloud_remove.detect_cloud import main_predict_cloud

def main(list_fp_img_selected, tmp_dir, FN_MODEL, sort_amount_of_clouds, first_image):
    print("Transform data to 0 -> 1...")
    dir_img_float_tmp = main_gendata(list_fp_img_selected, tmp_dir)
    print("Done")
    print("Predict cloud...")
    dir_predict_cloud_tmp = main_predict_cloud(dir_img_float_tmp, FN_MODEL)
    print("Done")
    print("Run remove cloud from image...")
    out_cloud, list_fn_sort = main_cloud_to_nodata(list_fp_img_selected, dir_predict_cloud_tmp, sort_amount_of_clouds, first_image)
    print("Done")
    return out_cloud, list_fn_sort

if __name__ == "__main__":
    workspace = '/home/quyet/data/aaaaa/'
    FN_MODEL = '/home/quyet/data/20211210_163952_0167_val_weights.h5'
    sort_amount_of_clouds = True
    first_image = None
    tmp_path = os.path.join(workspace, 'tmp')
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    list_folder = os.listdir(workspace)

    for i in list_folder:
        folder_path = os.path.join(workspace, i)
        list_month = glob.glob(os.path.join(folder_path, 'T*'))
        for j in list_month:
            list_fp_img_selected = glob.glob(j, '*.tif')
            name = i + '_' + os.path.basename(j)
            tmp_dir = os.path.join(tmp_path, name)
            out_fp = os.path.join(tmp_dir, name + ".tif")
            main(list_fp_img_selected, tmp_dir, out_fp, FN_MODEL, sort_amount_of_clouds, first_image, base_image=None)
    