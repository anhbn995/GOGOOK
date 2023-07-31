import glob, os
import multiprocessing

from osgeo import gdal

from multiprocessing.pool import Pool
from functools import partial


def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.tif"):
        list_id.append(file[:-4])
    return list_id

def crop_image(image_id, path_image_crop, folder_image, size_crop, true_size):
    filename = os.path.join(folder_image,image_id+'.tif')
    dataset_image = gdal.Open(filename)
    w,h = dataset_image.RasterXSize,dataset_image.RasterYSize
    list_hight_1 = list(range(0,h,true_size))
    list_weight_1 = list(range(0,w,true_size))
    list_hight = []
    list_weight = []
    for i in list_hight_1:
        if i < h - size_crop:
            list_hight.append(i)        
    list_hight.append(h-size_crop)
    
    for i in list_weight_1:
        if i < w - size_crop:
            list_weight.append(i)        
    list_weight.append(w-size_crop)
    
    count = 0
    for i in range(len(list_hight)):
        hight_tiles_up = list_hight[i]
        for j in range(len(list_weight)):
            weight_tiles_up = list_weight[j]
            count = count+1
            output_image = os.path.join(path_image_crop,r'%s_%s.tif'%(image_id,str('{0:04}'.format(count))))
            gdal.Translate(output_image, dataset_image,srcWin = [weight_tiles_up,hight_tiles_up,size_crop,size_crop])
    return True

def main_crop_overlap(folder_image, size_crop, stride):
    foder_name = os.path.basename(folder_image)
    size_crop = size_crop
    true_size = stride
    parent = os.path.dirname(folder_image)
    core = multiprocessing.cpu_count()
    if not os.path.exists(os.path.join(parent,foder_name+'_crop')):
        os.makedirs(os.path.join(parent,foder_name+'_crop'))
    path_image_crop = os.path.join(parent,foder_name+'_crop')

    list_id = create_list_id(folder_image)
    p_cnt = Pool(processes=core)
    p_cnt.map(partial(crop_image,path_image_crop=path_image_crop, folder_image=folder_image, 
                        size_crop=size_crop, true_size=true_size), list_id)
    p_cnt.close()
    p_cnt.join()

    return path_image_crop
