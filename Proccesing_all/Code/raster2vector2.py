import glob, os
import rasterio
import cv2
import sys
import numpy as np
import geojson
from osgeo import gdal, ogr, osr
from pyproj import Proj, transform
from shapely.geometry import Polygon
from osgeo import osr
import fiona
from shapely import geometry
from multiprocessing import Pool


def unique(list1): 
    x = np.array(list1) 
    return np.unique(x)

def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.tif"):
        list_id.append(file[:-4])
    return list_id

def read_mask(url):
    with rasterio.open(url) as src:
        array = src.read()[0]
    return array

def list_cnt_to_list_cnx(list_cnt):
    list_cnx =[]
    for i in range(len(list_cnt)):
        cnx = np.reshape(list_cnt[i], (len(list_cnt[i]),2))
        cnx = cnx.astype(int)
        list_cnx.append(cnx)
    return list_cnx

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

#im, mask, dien tich xoa => contour rm, mask _rm
def remove_area(base_path, mask_base, area):
    im2, contours, hierarchy = cv2.findContours(mask_base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour2 = []
    for cnt in contours:
        if cv2.contourArea(cnt) > area:
            contour2.append(cnt)

    with rasterio.open(base_path) as src:
        transform1 = src.transform
        w,h = src.width,src.height
    mask_remove = np.zeros((h,w), dtype=np.uint8)
    list_cnx = list_cnt_to_list_cnx(contour2)
    cv2.fillPoly(mask_remove, list_cnx, 255)
    return contour2, mask_remove  

def rm_small_area(base_path, mask_base, outputFileName, area, coordinate_img):
    contour2, mask_remove = remove_area(base_path, mask_base, area)

    with rasterio.open(base_path) as src:
        transform1 = src.transform
        w,h = src.width,src.height

    str_coordinate_img = 'epsg:' + str(coordinate_img) 
    crs = rasterio.crs.CRS({"init": str_coordinate_img})
    new_dataset = rasterio.open(outputFileName, 'w', driver='GTiff',
                                height = h, width = w,
                                count=1, dtype="uint8",
                                crs=crs,
                                transform=transform1,
                                compress='lzw')
    new_dataset.write(mask_remove,1)
    new_dataset.close()


def rm_small_area_all(path_img, path_mask, area, coordinate_img):
    id_img_list = create_list_id(path_img)
    # print(id_img_list)
    parent = os.path.dirname(path_img)
    foder_name = os.path.basename(path_img)    
    img_procesing = os.path.join(parent,foder_name+'_mask_remove_xong_xoa')    
    if not os.path.exists(img_procesing):
        os.makedirs(img_procesing)

    for id_img in id_img_list:
        img_path = os.path.join(path_img, id_img + '.tif')
        mask_path = os.path.join(path_mask, id_img + '.tif') 
        mask_base = read_mask(mask_path) 
        outputFileName = os.path.join(img_procesing, id_img + '_rm.tif')   
        rm_small_area(img_path, mask_base, outputFileName, area, coordinate_img)
    return parent, img_procesing


def raster2vecter(path_im_mask):
    
    # path_im_mask = r"/media/khoi/Image/India/Image4.tif"
    mask = read_mask(path_im_mask)


    im1, contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    #danh sach contour khong co lo
    list_contour_not_holes = []
    #danh sach cac contour co lo
    list_contour_holes = []
    parents = []
    for i in range(len(contours)):
        if hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0:
            list_contour_not_holes.append(contours[i])
        if hierarchy[0][i][3] > 0:
            parents.append(hierarchy[0][i][3])
            list_contour_holes.append(contours[i])                
    parents = unique(parents)

    #danh sach cac contour la cha, moi danh sach thi thang dau luon la cha
    list_list_contour_parent = []
n
    for i in range(len(parents)):
        contour_parent = [contours[parents[i]]]
        for j in range(len(contours)):
            if hierarchy[0][j][3] == parents[i]:
                contour_parent.append(contours[j])
        list_list_contour_parent.append(contour_parent)


    # chua 2 thu la list polygon khong co lo, va list cac contour co lo
    polygons_result = [list_contour_not_holes, list_list_contour_parent]
    
    return polygons_result

def raster2vecter_all(folder_img_path, folder_mask_path, area, coordinate):
    parent, img_procesing = rm_small_area_all(folder_img_path, folder_mask_path, area, coordinate)
    outputFolderNameShp = os.path.join(parent, 'shp_result') 

    id_img_list = create_list_id(img_procesing)
    if not os.path.exists(outputFolderNameShp):
        os.makedirs(outputFolderNameShp)

    for id_img in id_img_list:
        img_mask_path = os.path.join(img_procesing, id_img + '.tif')
        polygons_result = raster2vecter(img_mask_path)
        # print(img_mask_path)
        outputFileName = os.path.join(outputFolderNameShp, id_img + '_rs.shp')
        # print(outputFileName)

        # outputFileName = r"/media/khoi/Image/India/Image4.shp"

        dataset_base = gdal.Open(img_mask_path)
        driverName = "ESRI Shapefile"       
        geotransform = dataset_base.GetGeoTransform()
        projection = osr.SpatialReference(dataset_base.GetProjectionRef())
        export_shapefile(polygons_result, geotransform, projection, outputFileName, driverName)


def main():
    # Main
    # thong so su dung area = 800, coordinate chinhs laf toa do 3857
    img_path = os.path.abspath(sys.argv[1])
    mask_path = os.path.abspath(sys.argv[2])
    # thuong la 800
    area = int(sys.argv[3])
    # 3785
    coordinate = int(sys.argv[4])

    # rm_small_area_all(img_path,mask_path,area,coordinate)
    raster2vecter_all(img_path, mask_path, area, coordinate)

if __name__=="__main__":
    main()