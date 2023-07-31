import os, glob
import gdalnumeric
import rasterio
import rasterio.features
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

def get_list_fp(folder_dir, type_file = '*.tif'):
        """
            Get all file path with file type is type_file.
        """
        list_fp = []
        for file_ in glob.glob(os.path.join(folder_dir, type_file)):
            head, tail = os.path.split(file_)
            list_fp.append(os.path.join(head, tail))
        return list_fp


def create_mask_by_shapefile(gdf_shp, height, width, tr):
    # shape_file = os.path.expanduser(shape_path)
    # shp = gp.read_file(shape_file)
    sub_provine_names = gdf_shp['NAME_3']
    masks = []
    for i in range(len(gdf_shp)):
        row = gdf_shp.loc[[i]]
        mask = rasterio.features.rasterize(row['geometry'],
                                    out_shape=(height, width),
                                    transform=tr
                                    )
        masks.append(mask)
    return masks, sub_provine_names.tolist()


def cal_area_single_commune(mask_predict_commune, dict_value_object, resolution, don_vi="m"):
    if don_vi=="km":
        resolution = resolution/1000
    dict_object_value_count = {}
    for key_value in dict_value_object:
        count_object = (mask_predict_commune==key_value).sum()
        name_object = dict_value_object[key_value]
        dict_object_value_count.update({name_object: count_object*resolution*resolution})
    return dict_object_value_count


def cal_area_for_all_commune(maks_commues, communes, mask_predict, dict_value_object, resolution, don_vi="m"):
    dict_commne_object_value = {}
    for idx, mask_commune in enumerate(maks_commues):
        mask_predict_commune = mask_commune*mask_predict[0]
        dict_object_value = cal_area_single_commune(mask_predict_commune, dict_value_object, resolution, don_vi=don_vi)
        commnue_name = communes[idx]
        dict_commne_object_value.update({commnue_name: dict_object_value})
    return dict_commne_object_value


def export_statistic_to_csv(dict_satisfy, fp_csv_out):
    df_satisfy = pd.DataFrame.from_dict(dict_satisfy, orient='index')
    print(df_satisfy)
    df_satisfy.to_csv(fp_csv_out)


def statisticy_mono_img(fp_img, df_capxa, object_value, fp_out_csv, resolution=0.5, donvi='m'):
    src = rasterio.open(fp_img)
    mask_predict =  src.read()
    bounds  = src.bounds
    geom = box(*bounds)
    df_bound_img = gpd.GeoDataFrame({"id":1,"geometry":[geom]})
    df_intersection = gpd.overlay(df_capxa, df_bound_img, how='intersection')
    masks, communes = create_mask_by_shapefile(df_intersection, src.height, src.width, src.transform)
    df_statistic = cal_area_for_all_commune(masks, communes, mask_predict, object_value, resolution=resolution, don_vi=donvi)
    export_statistic_to_csv(df_statistic, fp_out_csv)
    
    
def main(fp_dir_out_csv, folder_dir, shp_capxa, object_value, resolution=0.5, donvi='m'):
    if not os.path.exists(fp_dir_out_csv):
        os.makedirs(fp_dir_out_csv)
        
    df_xa = gpd.read_file(shp_capxa)
    list_fp_img = get_list_fp(folder_dir, type_file = '*.tif')
    
    for fp in list_fp_img:
        name_file = os.path.basename(fp)
        fp_out_csv = os.path.join(fp_dir_out_csv, name_file[:-4] + '.csv')
        print(name_file)
        statisticy_mono_img(fp, df_xa, object_value, fp_out_csv, resolution=0.5, donvi='m')
        
        
    
if __name__ == "__main__":

    folder_dir=r"/home/skm/SKM/WORK/KhongGianXanh/Data/3_data_and_model/Data_uint8/tmp/Result" 
    fp_dir_out_csv = r"/home/skm/SKM/WORK/KhongGianXanh/Data/3_data_and_model/Data_uint8/tmp/Result/CSV" 
    shp_capxa=r"/home/skm/SKM/WORK/KhongGianXanh/Data/1_data_origin/COG/COG/CapXa.shp"
    object_value = {
                1:"Nuoc",
                2:"Co Bui",
                3:"Cay Co Tan"
                }
    main(fp_dir_out_csv, folder_dir, shp_capxa, object_value, resolution=0.5, donvi='m')
