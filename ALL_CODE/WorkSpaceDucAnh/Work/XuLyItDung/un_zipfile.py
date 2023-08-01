import zipfile
import glob
import os
from tqdm import tqdm

def get_list_name_file(path_folder, name_file = '*.zip'):
    list_img_dir = []
    for file_ in glob.glob(os.path.join(path_folder, name_file)):
        path, tail = os.path.split(file_)
        dirname = os.path.join(path, tail)
        list_img_dir.append(dirname)
    return list_img_dir

list_path_zip = get_list_name_file(r"E:\WORK\DEM_Super_Resolution")
for path_zip in tqdm(list_path_zip):
    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall(r"E:\WORK\DEM_Super_Resolution")
