# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 07:53:31 2018

@author: ducanh
"""
import multiprocessing
import rasterio
import numpy as np

import time

import os
import gdal
import cv2
from tqdm import *
import sys
import glob
import ogr
import osr
from lib.convert_datatype import list_contour_to_list_polygon, list_polygon_to_list_contour,contour_to_polygon
from lib.export_data import list_polygon_to_list_geopolygon,polygon_to_geopolygon
from shapely.geometry import Polygon
import fiona
from fiona.crs import from_epsg
from shapely import geometry
# K.clear_session()
def rm_polygon_err(list_polygon):
    list_poly_good =[]
    for polygon in list_polygon:
        if len(polygon) >= 3:
            list_poly_good.append(polygon)
    return list_poly_good
def export_shapefile(list_polygon, geotransform, outputFileName, driverName, coordinate,proj_str):

    # data
    list_contour_not_holes = list_polygon[0]
    list_list_contour_parent = list_polygon[1]

    # # # xử lý không có lỗ
    list_polygon_not_holes = []
    list_poly_not_holes = list_contour_to_list_polygon(list_contour_not_holes)
    list_poly_not_holes = rm_polygon_err(list_poly_not_holes)
    list_geopolygon_not_holes = list_polygon_to_list_geopolygon(list_poly_not_holes, geotransform)
    # print(list_geopolygon_not_holes)
    for geopolygon in list_geopolygon_not_holes:
        geopolygon_not_holes = list(geopolygon)
        myPoly = geometry.Polygon(geopolygon_not_holes)
        # print(myPoly)
        list_polygon_not_holes.append(myPoly)

    # xử lý có lỗ
    list_polygon_have_holes = []
    for list_contour_parent in list_list_contour_parent:
        # những thằng là cha
        contour_parents = list_contour_parent[0]
        poly_parents = contour_to_polygon(contour_parents)
        geopolygon_parent = polygon_to_geopolygon(poly_parents, geotransform)
        geopolygon_parent = list(geopolygon_parent)
        # print(geopolygon_parent)
        #những thăng là con
        list_contour_child = np.delete(list_contour_parent,0)
        list_contour_child = rm_polygon_err(list_contour_child)
        list_poly_child = list_contour_to_list_polygon(list_contour_child)
        list_geopolygon_child = list_polygon_to_list_geopolygon(list_poly_child, geotransform)
        # print(list_geopolygon_child)
        #tung geopolygon đươc cho vao 1 list
        geopolygon_child_list = []
        for geopolygon_child in list_geopolygon_child:
            geopolygon_child_list.append(list(geopolygon_child))
        # print(geopolygon_child_list)
        myPoly = geometry.Polygon(geopolygon_parent,geopolygon_child_list)
        list_polygon_have_holes.append(myPoly)

    list_polygon = list_polygon_not_holes + list_polygon_have_holes
    if coordinate != None:
        print("anh",coordinate)
        wgs84 = fiona.crs.from_epsg(coordinate)
    else:
        wgs84 = fiona.crs.from_string(proj_str)
    schema = {'geometry':'Polygon', 'properties': {'id': 'int'}}
    with fiona.open(outputFileName, 'w', crs=wgs84,driver='ESRI Shapefile',schema=schema) as c:
        for polygon in list_polygon:
            c.write({
                    'geometry': geometry.mapping(polygon),
                    'properties': {'id': 1}
                    }) 


def list_cnt_to_list_cnx(list_cnt):
    list_cnx =[]
    for i in range(len(list_cnt)):
    #    cnx = np.reshape(list_cnt[i], (1,len(list_cnt[i]),2))
        cnx = np.reshape(list_cnt[i], (len(list_cnt[i]),2))
        cnx = cnx.astype(int)
        list_cnx.append(cnx)
    return list_cnx

def unique(list1): 
    x = np.array(list1) 
    return np.unique(x)

def read_mask(url):
    with rasterio.open(url) as src:
        array = src.read()[0]
    return array

def raster_to_vector(base_path,outputFileName, area_remove=1500):

    mask_base = read_mask(base_path)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

    closing = cv2.morphologyEx(mask_base, cv2.MORPH_CLOSE, kernel3)
    for i in range(5):
        closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel2)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
    del closing
    for i in range(5):
        opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel2)
    input_img = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel3)

    im2, contours, hierarchy = cv2.findContours(input_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) 
    #danh sach contour khong co lo
    list_contour_not_holes = []
    #danh sach cac contour co lo
    list_contour_holes = []
    parents = []

    for i in range(len(contours)):
        if hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0 :
            if cv2.contourArea(contours[i]) > area_remove:
                list_contour_not_holes.append(contours[i])
        if hierarchy[0][i][3] > 0  :
            parents.append(hierarchy[0][i][3])
            list_contour_holes.append(contours[i])
            
    parents = unique(parents)


    #danh sach cac contour la cha, moi danh sach thi thang dau luon la cha
    list_list_contour_parent = []

    for i in range(len(parents)):
        contour_parent = [contours[parents[i]]]
        for j in range(len(contours)):
            if hierarchy[0][j][3] == parents[i]:
                contour_parent.append(contours[j])
        list_list_contour_parent.append(contour_parent)

    # chua 2 thu la list polygon khong co lo, va list cac contour co lo
    polygons_result = [list_contour_not_holes, list_list_contour_parent]

    dataset_base = gdal.Open(base_path)
    driverName = "ESRI Shapefile"
    outputFileName = outputFileName
    # # coordinate = 3785
    # proj = osr.SpatialReference(wkt=dataset_base.GetProjection())
    # coordinate = (proj.GetAttrValue('AUTHORITY',1))
    with rasterio.open(base_path, mode='r+') as src:
        projstr = (src.crs.to_proj4())
        crs = src.crs
        check_epsg = crs.is_epsg_code
        # if check_epsg:
        coordinate = src.crs.to_epsg()
        # else:
            # coordinate = None
    print(coordinate,check_epsg)
    # projstr = (proj.ExportToProj4())
    geotransform = dataset_base.GetGeoTransform()
    export_shapefile(polygons_result, geotransform, outputFileName, driverName, coordinate=coordinate,proj_str = projstr)

def sum_image(img_dir, out_path):
    import glob
    def get_list_name_file(path_folder, name_file = '*.tif'):
        list_img_dir = []
        for file_ in glob.glob(os.path.join(path_folder, name_file)):
            _, tail = os.path.split(file_)
            list_img_dir.append(tail)
        return list_img_dir

    def write_image(data, height, width, numband, crs, tr, out, type=None):
        """
            Export numpy array to image by rasterio.
        """
        if type:
            dtype_custom = type
            data = data.astype(dtype_custom)
        else:
            dtype_custom = data.dtype
        with rasterio.open(
                            out,
                            'w',
                            driver='GTiff',
                            height=height,
                            width=width,
                            count=numband,
                            dtype=dtype_custom,
                            crs=crs,
                            transform=tr,
                            # nodata=0,
                            ) as dst:
                                dst.write(data)


    list_name = get_list_name_file(img_dir)
    if len(list_name) > 1:
        path = os.path.join(img_dir, list_name[0])
        src = rasterio.open(path)
        mask = src.read()
        mask[mask==255] = 1
        for name in list_name:
            path = os.path.join(img_dir, name)
            mask1 = rasterio.open(path).read()
            mask1[mask1==255] = 1
            mask += mask1
            print(np.unique(mask))
        a = np.max(mask)
        mask[mask < a] = 0
        mask[mask==a] = 255
    else:
        path = os.path.join(img_dir, list_name[0])
        src = rasterio.open(path)
        mask = src.read()

    write_image(mask, src.height, src.width, 1, src.crs, src.transform, out_path) 





if __name__ == '__main__':
    for i in range(1, 13):
        name = f"T{i}"
        img_dir = f"Z:\DA\\2_GreenSpaceSing\A\T1-T12_scene_crop_by_AOI_CLRM\{name}\data_genorator_01\predict_float"
        out_path = f"Z:\DA\\2_GreenSpaceSing\A\T1-T12_scene_crop_by_AOI_CLRM\shp\{name}.tif"
        print(img_dir)
        print(out_path)
        sum_image(img_dir, out_path)
        out_shape = f"Z:\DA\\2_GreenSpaceSing\A\T1-T12_scene_crop_by_AOI_CLRM\shp\{name}.shp"
        raster_to_vector(out_path, out_shape, 0)
        