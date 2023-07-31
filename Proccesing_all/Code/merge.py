# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 07:53:31 2018

@author: ducanh
"""
from multiprocessing.pool import Pool
from functools import partial
import rasterio
import numpy as np

def read_mask(url):
    with rasterio.open(url) as src:
        array = src.read()[0]
    return array

def merge_layer(image,maskdirname,outputFileName):
    def create_list_id(maskdirname):
        listfile=[]
        for file1 in os.listdir(maskdirname):
            if file1.endswith(".tif"):
                listfile.append(os.path.join(maskdirname,file1))
        return listfile
    listfile = create_list_id(maskdirname)
    mask = read_mask(listfile[0])
    with rasterio.open(image) as src:
        transform1 = src.transform
        w,h = src.width,src.height
    for file_path in listfile[1:]:
        msk = read_mask(file_path)
        mask = cv2.bitwise_or(mask,msk)
    # datax = np.zeros((h,w), dtype=np.uint8)
    # list_cnx = list_cnt_to_list_cnx(contour2)
    # cv2.fillPoly(datax, list_cnx, 255)

    crs = rasterio.crs.CRS({"init": "epsg:3857"})
    new_dataset = rasterio.open(outputFileName, 'w', driver='GTiff',
                                height = h, width = w,
                                count=1, dtype="uint8",
                                crs=crs,
                                transform=transform1,
                                compress='lzw')
# print(masking(r"/media/building/building/data_source/tmp/Malaysia-jupem/image_mask/forest.tif")[0].shape)
    new_dataset.write(mask,1)
    new_dataset.close()

    

if __name__ == '__main__':
    # return True
    # predict_all(r"/media/khoi/Data1/Gap/forest/img/bhuvan11.tif",r"/media/khoi/Data1/Gap/forest/img/bhuvan11.shp")
    # predict_all_2(r"/mnt/D850AAB650AA9AB0/planet_3m/DDN_PLanet_3m.tif",r"/mnt/D850AAB650AA9AB0/planet_3m/DDN_PLanet_3m_rs_unet_size128new.tif")
    # test()
    merge_layer(r"/mnt/D850AAB650AA9AB0/planet_3m/DDN_PLanet_3m.tif",r"/mnt/D850AAB650AA9AB0/planet_3m/result",r"/mnt/D850AAB650AA9AB0/planet_3m/merge_all.tif")
