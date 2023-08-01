import os, glob
import rasterio
import numpy as np
import geopandas as gpd
import rasterio.features
from tqdm.notebook import tqdm

def create_mask_all(img_path, shp_path, list_name_object, property_name, out_path_mask, gen_mask_unique = False, value_unique = 255):
    """
        property_name: day la ten truong ma chua cac doi tuong
    """
    with rasterio.open(img_path) as src:
        meta = src.meta
        height, width = src.height, src.width
        tr = src.transform
        crs_img = src.crs
    df = gpd.read_file(shp_path)
    df['valu'] = 0
    # check epsg
    if df.crs.to_string() != crs_img.to_string():
        df = df.to_crs(epsg=str(crs_img.to_epsg()))

    # check so luong class
    list_object_in_field = np.unique(df[property_name])
    print(f'check {list_name_object} is subset {list_object_in_field}')
    if set(list_name_object).issubset(list_object_in_field):
        i = 0
        for class_name in list_name_object:
            i+=1
            if class_name in list_object_in_field:
                df.loc[df[property_name] == class_name, 'valu']= i
            else:
                continue

        shapes = df[['geometry', 'valu']]
        shapes = list(map(tuple, shapes.values))
        mask = rasterio.features.rasterize(shapes, out_shape=(height, width), transform=tr)
        meta.update({   'count': 1, 
                        'nodata': 0,
                        'dtype': 'uint8'
                        })
        if gen_mask_unique:
            mask[mask!=0] = 1
            mask = mask*value_unique
        with rasterio.open(out_path_mask, 'w', **meta) as dst:
            dst.write(np.array([mask]))   
    else:
        print('roi vao exception')
        print(list_object_in_field)
        print(list_name_object)
        pass

    
if __name__ =='__main__':
    img_path = r"E:\WORK\Change_detection_Dubai\Data\image\KHALIFASAT_JUN2020.tif"
    shp_path = r"E:\WORK\Change_detection_Dubai\Data\label\Training_Sample_V3.shp"
    out_dir = r"E:\WORK\Change_detection_Dubai\Data\mask_building_change"
    os.makedirs(out_dir, exist_ok=True)
    list_name_object = ['Building Demolition', 'New Building', 'Rooftop Change', 'Existing Building Extension']
    property_name = 'Chng_Type'

    
    out_path_mask = os.path.join(out_dir, os.path.basename(img_path))
    create_mask_all(img_path, shp_path, list_name_object, property_name, out_path_mask, gen_mask_01=True)



