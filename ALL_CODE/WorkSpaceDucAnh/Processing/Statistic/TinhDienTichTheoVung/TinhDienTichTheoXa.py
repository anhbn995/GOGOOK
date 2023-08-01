# -*- coding: utf-8 -*-
import os, glob
import geopandas as gp
import rasterio
import rasterio.features
import numpy as np
import pandas as pd
from pandas import DataFrame


def create_mask_by_shapefile(shape_path, height, width, tr):
    shape_file = os.path.expanduser(shape_path)
    shp = gp.read_file(shape_file)
    sub_provine_names = shp['NAME_3']
    print(sub_provine_names)
    masks = []
    for i in range(len(shp)):
        row = shp.loc[[i]]
        mask = rasterio.features.rasterize(row['geometry'],
                                    out_shape=(height, width),
                                    transform=tr
                                    )
        masks.append(mask)
    
    return masks, sub_provine_names.tolist()


def cal_area(classified_mask, resolution, name_attr):
    unique, counts = np.unique(classified_mask, return_counts=True)
    pixel_and_number = list(zip(unique, counts))
    list_area_attr = []
    if len(unique) != len(name_attr)+1:
        print('zzzzzzzz')
        name = np.arange(len(name_attr)+1)
        i = np.array(list(set(name) - set(name).intersection(unique)))
        counts = np.insert(counts,i-np.arange(len(i)), 0)
        unique = name
        pixel_and_number = list(zip(unique, counts))

    for i in range(0, len(name_attr)):
        area = (pixel_and_number[i+1][1]*resolution**2)
        print("area of {}: {} m2".format(name_attr[i], area))
        list_area_attr.append(area)
    return np.asarray(list_area_attr)


def cal_area_agri_idx_by_attr(list_provine_mask, classified,name_provine, resolution, name_attr):
    list_list_area_attr = []
    for j, provine_mask in enumerate(list_provine_mask):
        classied_provine = classified * provine_mask
        print('Provine {}'.format(name_provine[j])) 
        print(np.unique(classied_provine))
        list_area_attr_provine = cal_area(classied_provine, resolution, name_attr)
        list_list_area_attr.append(list_area_attr_provine)
    return list_list_area_attr 


def export_csv(list_list_area_attr, name_attr, sub_provine_names, out_path):
    df = DataFrame (list_list_area_attr, index = sub_provine_names,columns = name_attr)
    df['Đơn vị'] = pd.Series('m2', index=df.index)
    print(df)
    df.to_csv(out_path, encoding='utf-8')
    
    
def get_list_name_file(path_folder, name_file = '*.tif'):
    list_img_dir = []
    for file_ in glob.glob(os.path.join(path_folder, name_file)):
        _, tail = os.path.split(file_)
        list_img_dir.append(tail)
    return list_img_dir
    

if __name__ == '__main__':
    # shape_path = r"D:\Work\KhongGianXanh\ThongKe\tayninh\diagioihanhchinh\gadm36_VNM_3.shp"
    # folder_img_path = r"D:\Work\KhongGianXanh\ThongKe\tayninh\img"
    
    shape_path = r"C:\Users\SkyMap\Desktop\b\oke.shp"
    folder_img_path = r"C:\Users\SkyMap\Desktop\c\7_predict_4class_add_2018_2020"
    out_dir = r"C:\Users\SkyMap\Desktop\c\7_predict_4class_add_2018_2020\csv"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # name_attr = ['BG', 'Cây Xanh', 'Mặt Nước', 'Khác']
    name_attr = ['1. Rừng', '2. Thảm cỏ', '3. Nước', '4. Khác']
    
    list_name_file = get_list_name_file(folder_img_path, name_file='*.tif')
    for name in list_name_file:
        path_img = os.path.join(folder_img_path, name)
        with rasterio.open(path_img) as src:
            tr = src.transform
            width, height = src.width,src.height
            predict_mask = src.read()
        list_mask_provine, sub_provine_names = create_mask_by_shapefile(shape_path, height, width, tr)
        list_list_area_attr = cal_area_agri_idx_by_attr(list_mask_provine, predict_mask, sub_provine_names, 10, name_attr)
        out_path_csv = os.path.join(out_dir, name[:-3] + "csv")
        export_csv(list_list_area_attr, name_attr, sub_provine_names, out_path_csv)
