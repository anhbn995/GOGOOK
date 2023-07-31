import os
import glob

from download_and_processing_cloud.greencover_api.mosaic_standard.utils import reproject_image
from download_and_processing_cloud.greencover_api.crop_image.normalize_data import norm_data
from download_and_processing_cloud.greencover_api.crop_image.step1_crop_image import main_cut_img 


def standard_coord(img_path, num_band, base_path, crs):
    print("Standardize data...")
    name_folder_cut = os.path.basename(img_path)
    reproject_folder = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'reproject_img', name_folder_cut)
    if not os.path.exists(reproject_folder):
        os.makedirs(reproject_folder)
    for i in glob.glob(os.path.join(img_path, '*.tif')):
        output_file = os.path.join(reproject_folder, os.path.basename(i))
        if not os.path.exists(output_file):
            reproject_image(i, output_file)
    standard_data = norm_data(reproject_folder, num_band)
    print("Done")
    # return standard_data
    return reproject_folder, standard_data # TO DO

def crop_image_month(tmp_path, img_path, box_path, base_path, name, num_band, crs):
    out_crop_dir = os.path.join(tmp_path, name+'_cut')
    # print(os.path.join(input_path, name))
    print("Crop image %s..."%(name))
    if os.path.exists(out_crop_dir):
        # kiem tra so file trong thu muc crop co du ko 
        if len(glob.glob(out_crop_dir+'/*.tif')) == len(glob.glob(os.path.join(img_path, '*.tif'))): 
            print("Exist crop image folder %s "%(name+'_cut'))
            out_standard_dir = standard_coord(out_crop_dir, num_band, base_path, crs)
        # neu khong du so luong file trong thu muc crop thi chay lai crop anh
        else:
            print("Exist crop image folder %s but is empty"%(name+'_cut'))
            print("Crop image %s..."%(name))
            input_crop_path = os.path.join(tmp_path, name)
            out_crop_dir = main_cut_img(img_path, box_path, input_crop_path)
            out_standard_dir = standard_coord(out_crop_dir, num_band, base_path, crs)
    # neu khong ton tai thu muc crop thi chay crop anh va sau do chay cloud remove
    else:   
        print("Crop image folder isn't exists")
        # print("\n")
        input_crop_path = os.path.join(tmp_path, name)
        out_crop_dir = main_cut_img(img_path, box_path, input_crop_path)
        out_standard_dir = standard_coord(out_crop_dir, num_band, base_path, crs)
    return out_standard_dir