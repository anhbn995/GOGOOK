import os
import glob

from download_and_processing_cloud.greencover_api.forest import main_forest
from download_and_processing_cloud.greencover_api.green_cover import classification
from download_and_processing_cloud.greencover_api.mosaic_standard import mosaic_month
from download_and_processing_cloud.greencover_api.cloud_mask import create_mask_cloud
from download_and_processing_cloud.greencover_api.crop_image import crop_image_month
from download_and_processing_cloud.greencover_api.cloud_remove import cloud_remove_month
from download_and_processing_cloud.greencover_api.mosaic_standard.utils import check_base_img
from download_and_processing_cloud.greencover_api.mosaic_standard import mosaic_view


def check_gpu():
    import nvidia_smi
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    nvidia_smi.nvmlShutdown()
    print("Check GPU memory before predicting")
    print("Total:", info.total)
    print("Free:", info.free)
    if info.free/info.total < 0.9:
        raise Exception("GPU Memory is used for another process")
    print("GPU is ready to predict"*100)



def check_exists_foler_with_name(folder_path, name):
    out_folder = os.path.join(folder_path, name)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    return out_folder


def sorted_month_folder(img_ori_path):
    list_year = []
    dict_m_y = {}
    new_m_y = []
    list_id_folder = os.listdir(img_ori_path)
    for id_folder in list_id_folder:
        month, year = id_folder.split('-')
        month = int(month.replace('T', ''))
        year = int(year)
        list_year.append(year)
        dict_m_y.setdefault(year, []).append(month)

    for year in sorted(list_year):
        for month in sorted(dict_m_y[year]):
            item = 'T'+str(month)+'-'+str(year)
            if os.path.join(img_ori_path, item) not in new_m_y:
                new_m_y.append(os.path.join(img_ori_path, item))
    return new_m_y


def main(folder_paths, weight_path_cloud, weight_path_green, weight_path_water,
         weight_path_forest, num_band=4, use_box=True, crs='EPSG:4326'):
    box_path = check_exists_foler_with_name(folder_paths, 'box')
    base_folder_path = check_exists_foler_with_name(folder_paths, 'base')
    tmp_path = check_exists_foler_with_name(folder_paths, 'tmp')
    tmp_path1 = check_exists_foler_with_name(folder_paths, 'tmp1')
    results_path = check_exists_foler_with_name(folder_paths, 'results')
    img_ori_path = os.path.join(folder_paths, 'img_origin')
    cloud_tmp_dir = '/home/geoai/geoai_data_test2/mosaic'
    temp_folder_dir = '/home/geoai/geoai_data_test2/temp'
    if not os.path.exists(img_ori_path):
        raise Exception(
            "You don't have img_ori folder, please check path: %s" % (img_ori_path))
    list_month_folder = sorted_month_folder(img_ori_path)
    print(list_month_folder)
    first_base_image = check_base_img(box_path, base_folder_path, use_box)

    list_rmcloud_img = []
    for n, month_folder in enumerate(list_month_folder):
        name = os.path.basename(month_folder)
        # TO DO:
        threshhold = 0.7
        cloud_choose_1 = os.path.join(
            month_folder, 'cloud_<_%s%%' % str(threshhold*100))
        cloud_choose_2 = os.path.join(
            month_folder, 'cloud_>_%s%%' % str(threshhold*100))
        if len(glob.glob(os.path.join(cloud_choose_1, '*.tif'))) > 0:
            cloud_choose = cloud_choose_1
        else:
            cloud_choose = cloud_choose_2
            if len(glob.glob(os.path.join(cloud_choose, '*.tif'))) == 0:
                raise Exception("Image folder is empty, please check %s" % (
                    os.path.dirname(cloud_choose)))
        if not use_box:
            # out_crop_dir = month_folder
            out_crop_dir = cloud_choose  # TO DO
        else:
            # out_crop_dir = crop_image_month(tmp_path, month_folder, box_path, first_base_image, name, num_band, crs)
            # TO DO:
            out_crop_dir = crop_image_month(
                tmp_path, cloud_choose, box_path, first_base_image, name, num_band, crs)
            reproject_folder, standard_data = out_crop_dir

        print(out_crop_dir)
        # out_img_cloud, list_fn_sort = cloud_remove_month(out_crop_dir, tmp_path, weight_path_cloud, first_base_image, name)
        # TO DO:
        if reproject_folder:
            out_img_cloud, _ = cloud_remove_month(
                reproject_folder, tmp_path, weight_path_cloud, first_base_image, name)
            linh_percentile, _ = cloud_remove_month(
                standard_data, tmp_path1, weight_path_cloud, first_base_image, name)
            print("Create view image by linh's parameters...")
            results_img_view = os.path.join(
                results_path, 'view', name+'_view.tif')
            if not os.path.exists(results_img_view):
                list_img_tmp = glob.glob(
                    os.path.join(linh_percentile, '*.tif'))
                listt = []
                for i in list_img_tmp:
                    name_image = os.path.basename(i).replace('.tif', '')
                    haze_index = name_image.split('_')[-2]
                    listt.append([int(haze_index), name_image])
                list_fn_sort = []
                for j in sorted(listt):
                    list_fn_sort.append(j[-1]+'.tif')

                mosaic_view(results_path, results_img_view,
                            linh_percentile, name, list_fn_sort, first_base_image)
        else:
            out_img_cloud, _ = cloud_remove_month(
                reproject_folder, tmp_path, weight_path_cloud, first_base_image, name)

        results_img_pci = os.path.join(results_path, 'view', name+'.tif')
        results_img_gdal = os.path.join(results_path, 'mosaic', name+'.tif')
        if not os.path.exists(results_img_pci) and not os.path.exists(results_img_gdal):
            list_img_tmp = glob.glob(os.path.join(out_img_cloud, '*.tif'))
            listt = []
            for i in list_img_tmp:
                name_image = os.path.basename(i).replace('.tif', '')
                haze_index = name_image.split('_')[-2]
                listt.append([int(haze_index), name_image])
            list_fn_sort = []
            for j in sorted(listt):
                list_fn_sort.append(j[-1]+'.tif')

            pci_img, gdal_img, rm_cloud_img = mosaic_month(results_path, out_img_cloud, cloud_tmp_dir,
                                                           temp_folder_dir, name, list_fn_sort, first_base_image)

            shp_folder = os.path.join(results_path, 'cloud')
            create_mask_cloud(rm_cloud_img, gdal_img, shp_folder, name)
        else:
            gdal_img = results_img_gdal
        list_rmcloud_img.append(gdal_img)
        print("***** %s finished *****" % (name))
        print('\n')

    check_gpu()

    print("Run classification green cover...")
    result_classification = os.path.join(results_path, 'classification')
    classification(list_rmcloud_img, weight_path_green,
                   weight_path_water, result_classification)

    print("Run classification forest...")
    forest_folder = os.path.join(results_path, 'forest')
    main_forest(use_model='att_unet', weight_path=weight_path_forest,
                image_path=list_rmcloud_img, results_path=forest_folder)
    print("Finished")


if __name__ == "__main__":
    folder_paths = os.path.join(
        os.getcwd(), 'projects', 'Green Cover Npark Singapore')
    weight_path_cloud = os.path.join(
        os.getcwd(), 'weights', 'cloud_weights.h5')
    weight_path_green = os.path.join(
        os.getcwd(), 'weights', 'green_weights.h5')
    weight_path_water = os.path.join(
        os.getcwd(), 'weights', 'water_weights.h5')
    weight_path_forest = os.path.join(
        os.getcwd(), 'weights', 'forest_weights_v2.h5')
    # weight_path_green = '/home/quyet/WorkSpace/Model/Segmen_model/weig
    # ts/attunet_npark_greencover_stretch_128_1class_4_train.h5'
    # weight_path_water = '/home/quyet/WorkSpace/Model/Segmen_model/weights/unet3plus_water_npark_1_train.h5'
    main(folder_paths, weight_path_cloud, weight_path_green,
         weight_path_water, weight_path_forest)
