import os
import glob
from download_and_processing_cloud.greencover_api.mosaic_standard.step_3_mosaic_and_standard_img import mosaic_gdal
from download_and_processing_cloud.greencover_api.mosaic_standard.step_3_mosaic_and_standard_img import main_mosaic

def get_last_base(result_path, first_base_path, name):
    list_my = []
    list_resutl_folder = next(os.walk(result_path))[1]
    if len(list_resutl_folder) > 0:
        for my in  list_resutl_folder:
            result_gdal_path = os.path.join(result_path, my, 'DA', my+'_DA.tif') 
            result_pci_path = os.path.join(result_path, my, my+'.tif') 
            if os.path.exists(result_gdal_path) and os.path.exists(result_pci_path):
                month ,year = my.replace('T','').split('-')
                list_my.append([int(year), int(month)])
            elif os.path.exists(result_gdal_path) and not os.path.exists(result_pci_path):
                raise Exception("Result pci isn't exists but result gdal is exists, %s"%(result_pci_path))
            elif not os.path.exists(result_gdal_path) and os.path.exists(result_pci_path):
                raise Exception("Result gdal isn't exists but result pci is exists, %s"%(result_gdal_path))
            else:
                pass
        cur_month, cur_year = name.replace('T','').split('-')
        cur_month_year = [int(cur_year), int(cur_month)]
        if cur_month_year not in list_my:
            list_my.append(cur_month_year)
        arrange_list = sorted(list_my)
        index_cur = arrange_list.index(cur_month_year)
        if index_cur == 0:
            base_gdal_path = first_base_path
            base_pci_path = first_base_path
        else:
            pre_month_year = arrange_list[index_cur -1]
            str_pre_month_year = 'T'+str(pre_month_year[1])+'-'+str(pre_month_year[0])
            base_gdal_path = os.path.join(result_path, str_pre_month_year, 'DA', str_pre_month_year+'_DA.tif')
            base_pci_path = os.path.join(result_path, str_pre_month_year, str_pre_month_year+'.tif')
    else:
        base_gdal_path = first_base_path
        base_pci_path = first_base_path
    return base_gdal_path, base_pci_path

def mosaic_month(result_path, out_img_cloud, cloud_tmp_dir, temp_folder_dir, 
                name, list_img_name, first_base_path):
    folder_pci = os.path.join(result_path, 'view')
    folder_gdal = os.path.join(result_path, 'mosaic')
    if not os.path.exists(folder_pci):
        os.mkdir(folder_pci)
    if not os.path.exists(folder_gdal):
        os.mkdir(folder_gdal)

    results_img_pci = os.path.join(folder_pci, name+'.tif')
    results_img_gdal = os.path.join(folder_gdal, name+'.tif')
    
    base_path_gdal, base_path_pci = get_last_base(result_path, first_base_path, name)
    pci_img, gdal_img, rm_cloud_img = main_mosaic(results_img_pci, results_img_gdal, out_img_cloud, cloud_tmp_dir, 
                                                temp_folder_dir, name, list_img_name, base_path_gdal, base_path_pci)
    return pci_img, gdal_img, rm_cloud_img

def mosaic_view(result_path, results_img_pci, out_img_cloud, name, list_img_name, first_base_path):
    folder_gdal = os.path.join(result_path, 'view')
    if not os.path.exists(folder_gdal):
        os.mkdir(folder_gdal)

    # results_img_pci = os.path.join(folder_gdal, name+'.tif')    
    base_path_gdal, _ = get_last_base(result_path, first_base_path, name)
    
    base_to_convert = base_path_gdal
    remove_cloud_folder = os.path.join(os.path.dirname(out_img_cloud), 'cloud_mask')
    if not os.path.exists(remove_cloud_folder):
        os.mkdir(remove_cloud_folder)
    if not os.path.exists(remove_cloud_folder):
        os.mkdir(remove_cloud_folder)
    remove_cloud_img = os.path.join(remove_cloud_folder, name+'.tif')
    mosaic_gdal(out_img_cloud, list_img_name, remove_cloud_img, None, base_to_convert)

    remove_cloud_img = os.path.join(remove_cloud_folder, name+'.tif')
    gdal_img = mosaic_gdal(out_img_cloud, list_img_name, results_img_pci, base_path_gdal, remove_cloud_img)

    # pci_img, gdal_img, rm_cloud_img = main_mosaic(results_img_pci, , out_img_cloud, cloud_tmp_dir, 
    #                                             temp_folder_dir, name, list_img_name, base_path_gdal, base_path_pci)
    return gdal_img 