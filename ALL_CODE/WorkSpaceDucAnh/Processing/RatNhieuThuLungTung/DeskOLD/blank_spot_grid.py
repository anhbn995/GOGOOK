# -*- coding: utf-8 -*-
"""
Created on Tus Jul 15 07:49:23 2020

@author: DucAnh
"""

import os, glob
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
from scipy.spatial import Delaunay
from functools import partial 
from multiprocessing import Pool
import time
import argparse

def polygon_breath(poly):
    box = poly.minimum_rotated_rectangle
    x, y = box.exterior.coords.xy
    edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
    length = max(edge_length)
    return (poly.length)/length

def get_all_list_file_from_dir(img_cut_dir, name_file):
    list_img_dir = []
    for file_ in glob.glob(os.path.join(img_cut_dir, name_file)):
        # print(file_)
        list_img_dir.append(file_)
    return list_img_dir

def export_tri_delaunay(area, point_shp_path): 
    """
    Xuat ra file tam giac co the them cay
    
    Parameters
    ----------
    area : TYPE
        DESCRIPTION.
    point_shp_path : TYPE
        DESCRIPTION.

    Returns
    -------
    out_path_folder : TYPE
        noi chua shp tam giac.

    """    
    head,tail = os.path.split(point_shp_path) 
    out_path_folder = os.path.join(head,"_tmp")
    if not os.path.exists(out_path_folder):
        os.mkdir(out_path_folder)
    out_path_file = os.path.join(out_path_folder,tail[:-4]+".geojson")
    
    shp = gpd.read_file(point_shp_path)
    crs_shp = shp.crs#['init']

    lon = shp['geometry'].y
    lat = shp['geometry'].x
    lat_long =  pd.concat([lat, lon], axis=1, sort=False)
    all_point = lat_long.to_numpy()
    
    tri = Delaunay(all_point)
    geo = all_point[tri.simplices]
    
    convert_perimeter = lambda x: polygon_breath(Polygon(x))
    convert_AREA = lambda x: Polygon(x).area
    convert_shapely = lambda x: Polygon(x)
    
    numpy_perimeter = np.array(list(map(convert_perimeter,geo)))
    numpy_AREA = np.array(list(map(convert_AREA,geo)))
    numpy_shapely = np.array(list(map(convert_shapely,geo)))
    
    index_find_perimeter = np.where(numpy_perimeter < 2.2)
    index_find_area = np.where(numpy_AREA > area)
    
    a= index_find_area[0][~np.isin(index_find_area[0],index_find_perimeter[0])]
    numpy_shapely = numpy_shapely[(a)]
    
    dataset = pd.DataFrame({'geometry': numpy_shapely})
    dataset.index.name = 'idx'
    dataset.reset_index()
    gdf = gpd.GeoDataFrame(dataset, crs=crs_shp)
    gdf.to_file(out_path_file, driver='GeoJSON')
    return out_path_file

    """Nguyen"""        
def intersect_tri_with_aoi_non(shp_non_path, tri):
    shp_aoi_non = gpd.read_file(shp_non_path)
    crs_4326 = {'init': 'epsg:4326'}
    shp_aoi_non = gpd.GeoDataFrame(shp_aoi_non, crs=crs_4326)
    crs = {'init': 'epsg:32748'}
    shp_aoi_non = shp_aoi_non.to_crs(crs)

    tri_with_aoi = gpd.overlay(tri, shp_aoi_non, how='intersection')
    tri_with_aoi = tri_with_aoi[tri_with_aoi['geometry'].area > 1]
    
    id_tri = tri_with_aoi['idx']#.unique()   
    idx=id_tri.tolist()
    tri_non_intersec = tri[~tri['idx'].isin(idx)]

    return tri_non_intersec
    
def filter_grid_care(out_grid_path, aoi_care_path, grid_path, tri_path, shp_non_path = 'None'):
    # get name image
    _, tail = os.path.split(grid_path) 
    name_file = tail[:-4]

    # make folder out grid
    folfer_out_grid, _ = os.path.split(out_grid_path) 
    if not os.path.exists(folfer_out_grid):
        os.mkdir(folfer_out_grid)

    shp_aoi_care = gpd.read_file(aoi_care_path)
    tri = gpd.read_file(tri_path)
    grid = gpd.read_file(grid_path)
    
    polygon_aoi_care = shp_aoi_care.loc[shp_aoi_care['IMG_ID'] == name_file]
    polygon_aoi_care = polygon_aoi_care['geometry']
    crs_4326 = {'init': 'epsg:4326'}
    aoi_care = gpd.GeoDataFrame(polygon_aoi_care, crs=crs_4326)
    crs = {'init': 'epsg:32748'}
    aoi_care = aoi_care.to_crs(crs)
    tri = gpd.GeoDataFrame(tri)
    tri_with_aoi = tri
        
    """tri intersect with aoi"""
    if shp_non_path != 'None':
        tri_with_aoi = intersect_tri_with_aoi_non(shp_non_path, tri)  
        
    """tri intersect with grid"""
    tri_aoi_grid_intersec = gpd.overlay(tri_with_aoi, grid, how='intersection')
    k = tri_aoi_grid_intersec[tri_aoi_grid_intersec['geometry'].area > 4]
    id_grid_aoi = k['id'].unique()   
    b=id_grid_aoi.tolist()
    tri_aoi_grid_intersec = grid[grid['id'].isin(b)]
  
    """ grid intersect with aoi_care """
    grid_filter = gpd.GeoDataFrame(tri_aoi_grid_intersec, crs=crs)
    result_gpd = gpd.overlay(grid_filter, aoi_care, how='intersection')

    k = result_gpd[result_gpd['geometry'].area > 75]  
    id_grid_aoi = k['id'].unique()   
    b=id_grid_aoi.tolist()
    result_gpd = grid[grid['id'].isin(b)]    
    crs = {'init': 'epsg:4326'}
    gdf = result_gpd.to_crs(crs)
    gdf.to_file(out_grid_path)

def blank_sport_main(out_grid_path, point_shp_path, aoi_care_path, grid_path, shp_non_path = 'None', area = 7.5):
    tri_delaunay_path = export_tri_delaunay(area, point_shp_path)
    filter_grid_care(out_grid_path, aoi_care_path, grid_path, tri_delaunay_path, shp_non_path = 'None')

if __name__ == '__main__':
    area = 8.5 
    point_shp_path = r"E:\Blank_spot\point_rs\Version2\Acra_1708\Point\S01_202004021201_JB_DB_SPD0013600_RGB_point.shp"

    """Blinedtest_Acra_1708 32748"""
    aoi_care_path = r"E:\Blank_spot\Aoi_care\ACRA\Petak_Jambi_below40days_06_Petaks.shp"
    grid_path = r"E:\Blank_spot\Grid_origin\ACRA\S01_202004021201_JB_DB_SPD0013600_RGB.shp"
    shp_non_path = r"E:\Blank_spot\Non_bspot\ACRA\S01_202004021201_JB_DB_SPD0013600_RGB.shp"
    out_grid_path = r"C:\Users\skymap\Desktop\ThoNhuongDat\hihihihihihihihih.shp"
    blank_sport_main(out_grid_path, point_shp_path, aoi_care_path, grid_path, shp_non_path = 'None', area = 7.5)
    # args_parser = argparse.ArgumentParser()

    # args_parser.add_argument(
    #     '--tree_point_result',
    #     help='result tree detect point',
    #     required=True
    # )
    # args_parser.add_argument(
    #     '--aoi_care',
    #     help='boundary aoi',
    #     required=True
    # )
    # args_parser.add_argument(
    #     '--grid_dir',
    #     help='grid_result',
    #     required=True
    # )

    # args_parser.add_argument(
    #     '--non_blank_shape_dir',
    #     help='Non blank spot label',
    #     required=True
    # )
    # args_parser.add_argument(
    #     '--output_dir',
    #     help='foder result output',
    #     required=True
    # )
    # args_parser.add_argument(
    #     '--min_area_filter',
    #     help='min Area is blankspot',
    #     required=True
    # )
    # param = args_parser.parse_args()
    # folder_shp = param.tree_point_result
    # aoi_care_path1 = param.aoi_care
    # folder_grid_path = param.grid_dir
    # folder_shp_non_path = param.non_blank_shape_dir
    # out = param.output_dir
    # area = param.min_area_filter
    # filter_grid_care(out_folder_folder, aoi_care_path1, folder_shp_non_path, input_filter)
    