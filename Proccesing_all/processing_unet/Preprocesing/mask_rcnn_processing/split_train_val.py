import glob, os
from osgeo import gdal, gdalconst, ogr, osr
import numpy as np
import math
import sys
from pyproj import Proj, transform
foder_image = os.path.abspath(sys.argv[1])
# foder_image_mask = os.path.abspath(sys.argv[2])
foder_name = os.path.basename(foder_image)
parent = os.path.dirname(foder_image)
split = float(sys.argv[2])
def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.tif"):
        list_id.append(file[:-4])
        # print(file[:-4])
    return list_id

def main():
    import shutil
    image_list = create_list_id(foder_image)
    np.random.shuffle(image_list)
    count = len(image_list)
    cut_idx = int(round(count*split))
    print(cut_idx)
    train_list = image_list[0:cut_idx]

    # val_list = image_list[cut_idx:count]
    val_list = [id_image for id_image in image_list if id_image not in train_list]
    path_train = os.path.join(parent,foder_name+'_data','train','images')
    if not os.path.exists(path_train):
        os.makedirs(path_train)
    path_val = os.path.join(parent,foder_name+'_data','val','images')
    if not os.path.exists(path_val):
        os.makedirs(path_val)
    for image_name in train_list:
        # shutil.copy(os.path.join(foder_image,image_name+'.tif'), path_train)
        os.rename(os.path.join(foder_image,image_name+'.tif'), os.path.join(path_train,image_name+'.tif'))
    for image_name in val_list:
        # shutil.copy(os.path.join(foder_image,image_name+'.tif'), path_val)
        os.rename(os.path.join(foder_image,image_name+'.tif'), os.path.join(path_val,image_name+'.tif'))
if __name__=='__main__':
    main()
