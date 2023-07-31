import rasterio
import cv2
import numpy as np
import glob,os
import geojson
from osgeo import gdal, ogr, osr
from pyproj import Proj, transform
from shapely.geometry import Polygon
from osgeo import osr
import fiona
from shapely import geometry


def contour_to_polygon(contour):
    list_point = []
    for point in contour:
        [x,y] = point[0]
        point_in_polygon = (x,y)
        list_point.append(point_in_polygon)
    [x,y] = contour[0][0]
    point_in_polygon = (x,y)
    list_point.append(point_in_polygon)
    poly = tuple(list_point)
    return poly

def list_contour_to_list_polygon(list_contour):
    list_polygon = []
    for contour in list_contour:
        poly = contour_to_polygon(contour)
        list_polygon.append(poly)
    return list_polygon
def list_polygon_to_list_geopolygon(list_polygon, geotransform):
    list_geopolygon = []
    for polygon in list_polygon:
        geopolygon = polygon_to_geopolygon(polygon, geotransform)
        list_geopolygon.append(geopolygon)
    return list_geopolygon
def polygon_to_geopolygon(polygon, geotransform):
    temp_geopolygon = []
    for point in polygon:
        geopoint = point_to_geopoint(point, geotransform)
        temp_geopolygon.append(geopoint)
    geopolygon = tuple(temp_geopolygon)
    return geopolygon

def point_to_geopoint(point, geotransform):
    topleftX = geotransform[0]
    topleftY = geotransform[3]
    XRes = geotransform[1]
    YRes = geotransform[5]
    geopoint = (topleftX + point[0] * XRes, topleftY + point[1] * YRes)
    return geopoint

# def list_polygon_to_geopolygon(list_polygon,geotransform):
#     geopolygon = []
#     for polygon in list_polygon:
#         for point in polygon:
#             geopolygon.append(point)
#     return geopolygon

def rm_polygon_err(list_polygon):
    list_poly_good =[]
    for polygon in list_polygon:
        if len(polygon) >= 3:
            list_poly_good.append(polygon)
    return list_poly_good

def export_shapefile(list_polygon, geotransform, projection, outputFileName, driverName):

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
        print(myPoly)
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

    schema = {'geometry':'Polygon', 'properties': {'id': 'int'}}
    with fiona.open(outputFileName, 'w', 'ESRI Shapefile', schema) as c:
        for polygon in list_polygon:
            c.write({
                    'geometry': geometry.mapping(polygon),
                    'properties': {'id': 1}
                    })         

def unique(list1): 
    x = np.array(list1) 
    return np.unique(x)

def read_mask(base_path):
    with rasterio.open(base_path) as src:
        arr = src.read()[0]
    return arr


base_path = r"/media/skm/SKM/Czech/cezch_new/Img/predict_unit8/20200519_164432_val_weights_last/decidous19LESYM3_003_0021_rgb_epsg32633.tif"
mask = read_mask(base_path)


im1, contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

#danh sach contour khong co lo
list_contour_not_holes = []
#danh sach cac contour co lo
list_contour_holes = []
parents = []
for i in range(len(contours)):
    if hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0 :
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
    

dataset_base = gdal.Open(base_path)

driverName = "ESRI Shapefile"
outputFileName = r"/media/skm/SKM/Czech/cezch_new/Img/predict_unit8/20200519_164432_val_weights_last/decidous19LESYM3_003_0021_rgb_epsg32633.shp"
geotransform = dataset_base.GetGeoTransform()

projection = osr.SpatialReference(dataset_base.GetProjectionRef())

# chua 2 thu la list polygon khong co lo, va list cac contour co lo
polygons_result = [list_contour_not_holes, list_list_contour_parent]


export_shapefile(polygons_result, geotransform, projection, outputFileName, driverName)
