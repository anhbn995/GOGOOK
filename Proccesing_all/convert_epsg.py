import os, sys
import rasterio
import numpy as np
import time
from tqdm import tqdm

def create_folder(root_path,name_add):
    parent = os.path.dirname(root_path)
    foder_name = os.path.basename(root_path)
    if not os.path.exists(os.path.join(parent,foder_name+'_'+name_add)):
        os.makedirs(os.path.join(parent,foder_name+'_'+name_add))
    path_create = os.path.join(parent,foder_name+'_'+name_add)
    # print('created: {}'.format(path_create))
    return path_create

def create_list_file_and_out_folder(path, name_folder):
    # create foler noi chua anh moi luon
    out_folder = create_folder(path, name_folder)
    print("Created folder output: {}".format(out_folder))
    # het
    list_image = []
    for root, dirs, files in os.walk(path):
        # print(root)
        for file_ in files:
            if file_.endswith(".tif"):
                # list_image.append(os.path.join(root, file_))# trar veef list path
                list_image.append(file_)# trar veef list file
    return list_image, out_folder

def convert_epsg(image_path, out_path, out_epsg):
    crs_new = rasterio.crs.CRS({"init":"epsg:{}".format(out_epsg)})
    with rasterio.open(image_path) as src:
        image = src.read((1, 2, 3))
        tr = src.transform
        height = src.height
        width = src.width
        num_band = src.count
        dtypes = src.dtypes
    # print(dtypes[0])
    new_dataset = rasterio.open(out_path, 'w', driver='GTiff',
                        height = height, width = width,
                        count = num_band, dtype = dtypes[0],
                        crs = crs_new,
                        transform = tr,
                        nodata = 0,
                        compress='lzw')
    if num_band == 1:
        new_dataset.write(image, 1)
    else:
        for i in range(num_band):
            new_dataset.write(image[i],i+1)
    new_dataset.close()
    
def main():
    path_folder = os.path.abspath(sys.argv[1])
    epsg_new = int((sys.argv[2]))
    print("EPSG add: {}".format(epsg_new))
    list_img, out_folder = create_list_file_and_out_folder(path_folder, str(epsg_new))
    for name_file in tqdm(list_img):
        convert_epsg(os.path.join(path_folder, name_file), os.path.join(out_folder, name_file), epsg_new)

if __name__=="__main__":
    x1 = time.time()
    main()
    print(time.time() - x1, "second")






# chay khong dung abspath
# path_folder = r"/media/skm/Image/cezch/Data_Cezch/test"
# epsg_new = 32633

