# -*- coding: utf-8 -*-
from osgeo import gdal, gdalconst, ogr, osr
import numpy as np
import shapefile as shp
from pyproj import Proj, transform
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib.colors import rgb_to_hsv
from math import pi
import glob, os
import sys
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
import time
import argparse
core = multiprocessing.cpu_count()

def list_array_to_contour(list_array):
    contour = np.asarray(list_array,dtype=np.float32)
    contour_rs = contour.reshape(len(contour),1,2)
    return contour_rs

def list_list_array_to_list_contour(list_list_array):
    list_contour = []
    for list_array in list_list_array:
        contour = list_array_to_contour(list_array)
        list_contour.append(contour)
    return list_contour

def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.tif"):
        list_id.append(file[:-4])
        # print(file[:-4])
    return list_id
def create_list_id2(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.shp"):
        list_id.append(file[:-4])
        # print(file[:-4])
    return list_id
def epsr1_to_epsr2(epsr1,epsr2,shapes_list):
    inProj = Proj(init='epsg:%s'%(epsr1))
    outProj = Proj(init='epsg:%s'%(epsr2))
    list_list_point_convert = []
    for shapes in shapes_list:
        list_point=[]
        for point in shapes:
            # print(point)
            long,lat = point[0],point[1]
            x,y = transform(inProj,outProj,long,lat)
            list_point.append((x,y))
        list_list_point_convert.append(list_point)
    # print(list_list_point_convert)
    return list_list_point_convert
def get_all_polygon(shape_filepath, shape_list):
    my_shapes_list_ring0 = []
    my_shapes_list_ring_oth = []
    for shape_name in shape_list:
        sf=shp.Reader(os.path.join(shape_filepath,shape_name+'.shp'))
        my_shapes=sf.shapes()
        # print(my_shapes)
        shapes_lamda = list(map(lambda shape:list(shape.__geo_interface__["coordinates"]), my_shapes))
        print(shapes_lamda)
        if len(shapes_lamda)>0:
            if type(shapes_lamda[0])==list:
                shapes_list_ring0 = [list_shape_poly[0] for list_shape_poly in shapes_lamda]
                shapes_list_ring_oth = []
                for list_shape_poly in shapes_lamda:
                    for poly in list_shape_poly[1:]:
                        shapes_list_ring_oth.append(poly)
            else:
                shapes_list_ring_oth = list(shapes_lamda[1:])
                shapes_list_ring0 = list(shapes_lamda[0:1])
        else:
            shapes_list_ring0 = []
            shapes_list_ring_oth = []
        if shape_name=="Road_A_07":
            print(shapes_list_ring0)
        driverName = "ESRI Shapefile"
        driver = ogr.GetDriverByName(driverName)
        dataSource = driver.Open(os.path.join(shape_filepath,shape_name+'.shp'), 0)
        layer = dataSource.GetLayer()
        crs = layer.GetSpatialRef()
        # epsr1 =  crs.GetAttrValue('AUTHORITY',1)
        epsr1 = 4326
        list_list_point_convert_ring0 = epsr1_to_epsr2(epsr1,4326,shapes_list_ring0)
        my_shapes_list_ring0.extend(list_list_point_convert_ring0)
        # if shape_name=="Road_A_06":
        #     # print(list_list_point_convert_ring0)
        list_list_point_convert_ring_oth = epsr1_to_epsr2(epsr1,4326,shapes_list_ring_oth)
        my_shapes_list_ring_oth.extend(list_list_point_convert_ring_oth)
        # print(shapes_list_ring0)
    # print(my_shapes_list_ring_oth)
    return my_shapes_list_ring0,my_shapes_list_ring_oth
def geo_polygon_to_pixel_polygon(transformmer,list_list_point_convert):
    xOrigin = transformmer[0]
    yOrigin = transformmer[3]
    pixelWidth = transformmer[1]
    pixelHeight = -transformmer[5]

    list_list_point=[]
    for points_list in list_list_point_convert :
    #    i=i+1
        lis_poly=[]
        for point in points_list:
            # print(point)
            col = int((point[0] - xOrigin) / pixelWidth)
            row = int((yOrigin - point[1] ) / pixelHeight)
            lis_poly.append([col,row])
        lis_poly = np.asarray(lis_poly,dtype = np.int)
        list_list_point.append(lis_poly)
    return list_list_point
def list_cnt_to_list_cnx(list_cnt):
    list_cnx =[]
    for i in range(len(list_cnt)):
    #    cnx = np.reshape(list_cnt[i], (1,len(list_cnt[i]),2))
        cnx = np.reshape(list_cnt[i], (len(list_cnt[i]),2))
        cnx = cnx.astype(int)
        list_cnx.append(cnx)
    return list_cnx
def create_mask(image_id, img_dir, path_create,my_shapes_list_ring0,my_shapes_list_ring_oth):
    print(1)
    "đọc ảnh"
    driver = gdal.GetDriverByName('GTiff')
    filename = os.path.join(img_dir,image_id+'.tif')    
    dataset = gdal.Open(filename)
    proj = osr.SpatialReference(wkt=dataset.GetProjection())    
    epsr2 = (proj.GetAttrValue('AUTHORITY',1))
    # epsr2 = 4326
    "chuyen toa do "
    print(epsr2)

    list_list_point_cv_ring0 = epsr1_to_epsr2(4326,epsr2,my_shapes_list_ring0)
    print(list_list_point_cv_ring0)
    list_list_point_cv_ring_oth = epsr1_to_epsr2(4326,epsr2,my_shapes_list_ring_oth)

    data = dataset.ReadAsArray()
    img = np.array(data).swapaxes(0,1).swapaxes(1,2)
    "chuyen sang toa do pixel"
    transformmer = dataset.GetGeoTransform()
    list_list_point_ring0 = geo_polygon_to_pixel_polygon(transformmer,list_list_point_cv_ring0)
    list_list_point_ring_oth = geo_polygon_to_pixel_polygon(transformmer,list_list_point_cv_ring_oth)

    list_cnt_ring0 = list_list_array_to_list_contour(list_list_point_ring0)
    list_cnt_ring_oth =  list_list_array_to_list_contour(list_list_point_ring_oth)

    datax = np.zeros((dataset.RasterYSize,dataset.RasterXSize), dtype=np.uint8)

    list_cnx_ring0 = list_cnt_to_list_cnx(list_cnt_ring0)
    list_cnx_ring_other = list_cnt_to_list_cnx(list_cnt_ring_oth)

    ignore_mask_color = 1
    ignore_mask_color_2 = 0
    #cnx = np.reshape(list_cnt[3], (1,len(list_cnt[3]),2))
    #cv2.fillPoly(datax, cnx, ignore_mask_color)

    #    if (cnx.min()>0 and cnx.max()<4000):
    #    cv2.fillPoly(datax, cnx, ignore_mask_color)
    # cv2.fillPoly(datax, list_cnx_ring0, ignore_mask_color)
    for cnxx in list_cnx_ring0:
        cv2.fillPoly(datax, [cnxx], ignore_mask_color)

    cv2.fillPoly(datax, list_cnx_ring_other, ignore_mask_color_2)

    """Đoạn này thêm vào để khi sinh mask ngược lại nếu không cần thì comment"""
    # datax = 1 - datax
    """Kết thúc sinh mask"""

    output_mask =  os.path.join(path_create,image_id+'.tif')
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(output_mask,dataset.RasterXSize,dataset.RasterYSize,1,gdal.GDT_Byte)#gdal.GDT_Byte/GDT_UInt16
    dst_ds.GetRasterBand(1).WriteArray(datax)
    dst_ds.GetRasterBand(1).ComputeStatistics(False)
    dst_ds.SetProjection(dataset.GetProjection())
    dst_ds.SetGeoTransform(dataset.GetGeoTransform())
    dst_ds.FlushCache()
    return True


def main(img_dir, shape_path):    
    parent = os.path.dirname(img_dir)
    foder_name = os.path.basename(img_dir)

    list_id = create_list_id(img_dir)
    list_shape = create_list_id2(shape_path)
    if not os.path.exists(os.path.join(parent,foder_name+'_mask')):
        os.makedirs(os.path.join(parent,foder_name+'_mask'))
    path_create = os.path.join(parent,foder_name+'_mask')
    my_shapes_list_ring0,my_shapes_list_ring_oth = get_all_polygon(shape_path, list_shape)
    p_cnt = Pool(processes=core)
    result = p_cnt.map(partial(create_mask,img_dir=img_dir,path_create=path_create,my_shapes_list_ring0=my_shapes_list_ring0,my_shapes_list_ring_oth=my_shapes_list_ring_oth), list_id)
    p_cnt.close()
    p_cnt.join()
    return path_create

if __name__ == "__main__":
    # args_parser = argparse.ArgumentParser()

    # args_parser.add_argument(
    #     '--img_dir',
    #     help='Orginal Image Directory',
    #     required=True
    # )


    # args_parser.add_argument(
    #     '--shape_dir',
    #     help='Box cut directory',
    #     required=True
    # )

    # param = args_parser.parse_args()
    # img_dir = param.img_dir
    # shape_path = param.shape_dir

    img_dir =os.path.abspath(sys.argv[1])
    shape_path =os.path.abspath(sys.argv[2])

    x1 = time.time()
    main(img_dir,shape_path)
    print(time.time() - x1, "second")
