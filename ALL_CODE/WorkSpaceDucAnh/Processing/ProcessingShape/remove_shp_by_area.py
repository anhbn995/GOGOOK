import geopandas as gpd
import os, glob
from tqdm import tqdm

def remove_polygon_by_area(fp_shp, fp_out_shp, area):
    df_in = gpd.read_file(fp_shp)
    df_out = df_in[df_in.area > area]
    df_out.to_file(fp_out_shp)

if __name__=='__main__':
    area = 1501
    dir_need_remove = r"E:\TMP_XOA\shp_change_detection"
    dir_out = dir_need_remove + f'remove_{area}'
    os.makedirs(dir_out, exist_ok=True)

    list_fp = glob.glob(os.path.join(dir_need_remove, '*.shp'))
    for fp_in in tqdm(list_fp, desc=f'Run remove area < {area}'):
        fp_out = os.path.join(dir_out, os.path.basename(fp_in))
        remove_polygon_by_area(fp_in, fp_out, area)