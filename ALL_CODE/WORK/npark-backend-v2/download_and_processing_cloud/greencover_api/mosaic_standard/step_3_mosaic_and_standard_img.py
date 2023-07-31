import os
import shutil
import rasterio

from rasterio.merge import merge
from download_and_processing_cloud.greencover_api.mosaic_standard.mosaic_pci import main as merge_mosaic
from download_and_processing_cloud.greencover_api.mosaic_standard.utils import convert_profile, write_image

def mosaic_pci(results_img_pci, out_img_cloud, cloud_tmp_dir, temp_folder_dir, name, base_path):
    if not os.path.exists(cloud_tmp_dir):
        os.mkdir(cloud_tmp_dir)
    cloud_tmp_geotest= os.path.join(cloud_tmp_dir, name)
    if not os.path.exists(results_img_pci):
        out_merge_mosaic = merge_mosaic(out_img_cloud, cloud_tmp_geotest, temp_folder_dir, base_path, name)
        shutil.copyfile(out_merge_mosaic, results_img_pci)
        img_temp_PCI = out_merge_mosaic.replace('.tif', '_new.tif')
        if base_path:
            convert_profile(out_merge_mosaic, base_path, img_temp_PCI)
            os.remove(out_merge_mosaic)
            shutil.copyfile(img_temp_PCI, out_merge_mosaic)
            os.remove(img_temp_PCI)
        shutil.rmtree(cloud_tmp_geotest)
    return results_img_pci

def mosaic_gdal(dir_path, list_img_name, out_path, base_path, base_to_convert=None):
    src_files_to_mosaic = []
    if not os.path.exists(out_path):
        for name_f in list_img_name:
            fp = os.path.join(dir_path, name_f)
            src = rasterio.open(fp)
            src_files_to_mosaic.append(src)
        if base_path:
            src_files_to_mosaic.append(rasterio.open(base_path))
        mosaic, out_trans = merge(src_files_to_mosaic)
        write_image(mosaic, mosaic.shape[1], mosaic.shape[2], mosaic.shape[0], src.crs, out_trans, out_path)
        img_temp_gdal = out_path.replace('.tif', '_new.tif')
        if base_to_convert:
            base_path = base_to_convert

        if base_path:
            convert_profile(out_path, base_path, img_temp_gdal)
            os.remove(out_path)
            shutil.copyfile(img_temp_gdal, out_path)
            os.remove(img_temp_gdal)
    return out_path

def main_mosaic(results_img_pci, results_img_gdal, out_img_cloud, cloud_tmp_dir, 
                temp_folder_dir, name, list_img_name, base_path_gdal, base_path_pci):
    # mosaic pci final result
    print("Mosaic image with PCI and convert profile...")
    pci_img = mosaic_pci(results_img_pci, out_img_cloud, cloud_tmp_dir, temp_folder_dir, name, base_path_pci)
    print("Done")
    # mosaic gdal remove cloud result
    print("Mosaic image with gdal and convert profile...")
    if base_path_gdal == None:
        base_path_gdal = pci_img
    base_to_convert = base_path_gdal
    remove_cloud_folder = os.path.join(os.path.dirname(out_img_cloud), 'cloud_mask')
    if not os.path.exists(remove_cloud_folder):
        os.mkdir(remove_cloud_folder)
    remove_cloud_img = os.path.join(remove_cloud_folder, name+'.tif')
    mosaic_gdal(out_img_cloud, list_img_name, remove_cloud_img, None, base_to_convert)
    # mosaic gdal final result
    print("Base img: %s"%(base_path_gdal))
    print("List gdal mosaic: %s"%str(list_img_name)) # TO DO
    gdal_img = mosaic_gdal(out_img_cloud, list_img_name, results_img_gdal, base_path_gdal, remove_cloud_img)
    print("Done")
    return pci_img, gdal_img, remove_cloud_img
