# -*- coding: utf-8 -*-
import gdal
import ogr
import osr
import numpy as np
from pyproj import Proj, transform
import cv2
import time
from lib.export_data import exportResult2 as exportResult2
from step_2_find_building_contours.find_contour_controller import find_contour_main
from step_3_determinate_corners.determinate_corner_controller import determinate_corner_main
from step_3_determinate_corners.minE_reduced_search import reduce_search
from step_3_determinate_corners.minE_full_search import full_search
from step_5_correcting_footprints.correcting_controller import correcting_main
from step_4_adjusting_operators.adjusting_operator_controller import adjusting_operators_main
from lib.convert_datatype import list_contour_to_list_polygon, list_polygon_to_list_contour
import argparse
import json


def list_array_to_contour(list_array):
    contour = np.asarray(list_array, dtype=np.float32)
    contour_rs = contour.reshape(len(contour), 1, 2)
    return contour_rs


def list_list_array_to_list_contour(list_list_array):
    list_contour = []
    for list_array in list_list_array:
        contour = list_array_to_contour(list_array)
        list_contour.append(contour)
    return list_contour


def list_cnt_to_list_cnx(list_cnt):
    list_cnx = []
    for i in range(len(list_cnt)):
        #    cnx = np.reshape(list_cnt[i], (1,len(list_cnt[i]),2))
        cnx = np.reshape(list_cnt[i], (len(list_cnt[i]), 2))
        cnx = cnx.astype(int)
        list_cnx.append(cnx)
    return list_cnx


def en_scale(list_cnt, scale):
    list_new = []
    ignore_mask_color = 1
    for i in range(len(list_cnt)):
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        cnx = np.reshape(list_cnt[i], (len(list_cnt[i]), 2))
        cnx = cnx.astype(int)
        # print(list_cnt[i])
        mask = cv2.fillPoly(mask, [cnx], ignore_mask_color)
        mask_new = cv2.resize(
            mask, (int(1024//scale), int(1024//scale)), interpolation=cv2.INTER_CUBIC)
        mask_new = np.array((mask_new > 0.5).astype(np.uint8))
        _, contours, _ = cv2.findContours(
            mask_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnx = list_cnt[i].astype(np.float32)/scale
        # list_new.append(cnx)
        list_new.append(contours[0])
    return list_new


def de_scale(list_cnt, scale):
    list_new = []
    for i in range(len(list_cnt)):
        cnx = list_cnt[i].astype(np.float32)*scale
        list_new.append(cnx)
    return list_new


def remove_nan_contour(list_cnt):
    list_new = []
    for cnt in list_cnt:
        if not np.isnan(cnt).any():
            list_new.append(cnt)
    return list_new


def en_contour_origin(list_contour):
    list_translative = []
    list_en_contour = []
    for contour in list_contour:
        a = np.amin(contour, axis=0)
        list_translative.append(a)
        list_en_contour.append(contour-a)
    return list_en_contour, list_translative


def de_contour_origin(list_contour, list_translative):
    list_de_contour = []
    for i in range(len(list_contour)):
        list_de_contour.append(list_contour[i]+list_translative[i])
    return list_de_contour


def create_mask(shape_path, img_url, shp_out, on_processing=None, scale=1.0):    # print(1)
    # "đọc shapefile"
    # sf=shp.Reader(shape_path)
    # my_shapes=sf.shapes()
    # my_shapes_list = list(map(lambda shape: shape.points, my_shapes))
    # get epsg
    # with open(shape_path) as f:
    #     data = json.load(f)

    # my_shapes_list = [feature['geometry']['coordinates'][0] for feature in data['features']]
    # # print(my_shapes_list)
    # driverName = "GeoJSON"
    # if on_processing: on_processing(0.01)
    # driver = ogr.GetDriverByName(driverName)
    # dataSource = driver.Open(shape_path, 0)
    # layer = dataSource.GetLayer()



# def create_mask(shape_path, shape_out, image_url=None, on_processing=None, scale=1.0):
    # with open(shape_path) as f:
    #     data = json.load(f)

    # my_shapes_list = [feature['geometry']['coordinates'][0] for feature in data['features']]
    # src_projection = osr.SpatialReference()
    # src_projection.ImportFromEPSG(4326)




    import shapefile as shp
    sf = shp.Reader(shape_path)
    my_shapes = sf.shapes()
    my_shapes_list = list(map(lambda shape: shape.points, my_shapes))
    # print(my_shapes_list[0])
    # get epsg
    driverName = "ESRI Shapefile"
    driver = ogr.GetDriverByName(driverName)
    dataSource = driver.Open(shape_path, 0)
    layer = dataSource.GetLayer()
    crs = layer.GetSpatialRef()
    # print(crs)

    "đọc ảnh"
    driver = gdal.GetDriverByName('GTiff')
    # print(filename)
    dataset = gdal.Open(img_url)
    proj = osr.SpatialReference(wkt=dataset.GetProjection())
    if on_processing:
        on_processing(0.1)

    "chuyen toa do "

    tar_projection = osr.SpatialReference(wkt=dataset.GetProjection())
    src_projection = osr.SpatialReference()
    src_projection.ImportFromEPSG(4326)
    # src_projection.ImportFromEPSG(23856)
    wgs84_trasformation = osr.CoordinateTransformation(
        src_projection, tar_projection)

    list_list_point_convert = []
    for shapes in my_shapes_list:
        list_point = []
        for _point in shapes:
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(_point[0], _point[1])
            point.Transform(wgs84_trasformation)
            # print(point)
            # import time
            # time.sleep(50)
            list_point.append((point.GetX(), point.GetY()))
        list_list_point_convert.append(list_point)

    "chuyen sang toa do pixel"
    transformmer = dataset.GetGeoTransform()
    if on_processing:
        on_processing(0.15)
    xOrigin = transformmer[0]
    yOrigin = transformmer[3]
    pixelWidth = transformmer[1]
    pixelHeight = -transformmer[5]
    if on_processing:
        on_processing(0.05)
    list_list_point = []
    print("hello")
    for points_list in list_list_point_convert:
        #    i=i+1
        lis_poly = []
        for point in points_list:
            col = round((point[0] - xOrigin) / pixelWidth)
            row = round((yOrigin - point[1]) / pixelHeight)
            lis_poly.append([col, row])
        lis_poly = np.asarray(lis_poly, dtype=np.int)
        list_list_point.append(lis_poly)
    # print(list_list_point)
    print("xong for")
    list_cnt = list_list_array_to_list_contour(list_list_point)
    if on_processing:
        on_processing(0.2)
    # list_cnt = list_cnt.sort(key=len,reverse=True)
    list_cnt = sorted(list_cnt, key=len, reverse=True)
    list_cnt_en_translative, list_translative = en_contour_origin(list_cnt)
    list_cnt_en_translative = en_scale(list_cnt_en_translative, scale)
    x = time.time()
    (polygon_mine, linear_polygon) = determinate_corner_main(
        list_cnt_en_translative, reduce_search)
    if on_processing:
        on_processing(0.3)
    # print(linear_polygon)
    print(time.time() - x, "second")
    # print(polygon_mine[0],linear_polygon[0])
    all_fixed_contour, list_translative_new = adjusting_operators_main(
        linear_polygon, list_translative)  # linear_polygon
    list_result_contour = correcting_main(all_fixed_contour)
    if on_processing:
        on_processing(0.4)
    list_result_contour = correcting_main(list_result_contour)
    if on_processing:
        on_processing(0.5)
    list_result_contour = de_scale(list_result_contour, scale)
    if on_processing:
        on_processing(0.6)
    list_result_contour = de_contour_origin(
        list_result_contour, list_translative_new)
    if on_processing:
        on_processing(0.7)
    list_result_contour = remove_nan_contour(list_result_contour)
    if on_processing:
        on_processing(0.8)
    # print(np.isnan(list_result_contour[0]))
    polygons_result = list_contour_to_list_polygon(list_result_contour)
    if on_processing:
        on_processing(0.9)
    # print('okkkkkkkkkkkkkk')
    print("Bat dau luu shape file ")
    driverName = "ESRI Shapefile"
    # shape_file_path = os.path.join(path_shape_foder,id_image+'.shp')
    outputFileName = shp_out
    geotransform = dataset.GetGeoTransform()
    projection = osr.SpatialReference(dataset.GetProjectionRef())
    # print(projection)
    exportResult2(polygons_result, geotransform,
                  projection, outputFileName, driverName)

    # driverName = "GeoJSON"
    # dataset_image = gdal.Open(img_url)
    # geotransform = dataset_image.GetGeoTransform()

    # projection = osr.SpatialReference(dataset_image.GetProjectionRef())

    # exportResult2(polygons_result, geotransform, projection, shp_out, driverName)
    if on_processing:
        on_processing(1)


if __name__ == '__main__':
    # return True
    # args_parser = argparse.ArgumentParser()

    # args_parser.add_argument(
    #     '--img_dir',
    #     help='Orginal Image Directory',
    #     required=True
    # )

    # args_parser.add_argument(
    #     '--shape_in',
    #     help='shape file input',
    #     required=True
    # )
    # args_parser.add_argument(
    #     '--shape_out',
    #     help='shape file output',
    #     required=True
    # )
    # param = args_parser.parse_args()
    # shape_in = param.shape_in
    # shp_out = param.shape_out
    # img_url = param.img_dir
    """NUNUKAN_1"""
    # shape_in = r'/home/skm/SKM/WORK/Buiding/BIG/IMG_BIG_V2_NUNUKAN_1/Data_Train_and_Model/1_img_resize_4326/BF_BIG_NUNUKAN_03r_512s_400c/NUNUKAN_1.shp'
    # shp_out = r'/home/skm/SKM/WORK/Buiding/BIG/IMG_BIG_V2_NUNUKAN_1/Data_Train_and_Model/1_img_resize_4326/BF_BIG_NUNUKAN_03r_512s_400c/rectify/NUNUKAN_1_rectify.shp'
    # img_url = r'/home/skm/SKM/WORK/Buiding/BIG/IMG_BIG_V2_NUNUKAN_1/Data_Train_and_Model/1_img_resize_4326/NUNUKAN_1.tif'
    # shape_in = r'/mnt/66A8E45DA8E42CED/Dubai_Demo_Data_Preparation/Car/Image/predict_car/20190919_095020_ssc10_u0001_pansharpened.shp'
    # shp_out = r'/mnt/66A8E45DA8E42CED/Dubai_Demo_Data_Preparation/Car/Image/rectify/20190919_095020_ssc10_u0001_pansharpened.shp'
    # img_url = r'/mnt/66A8E45DA8E42CED/Dubai_Demo_Data_Preparation/Car/Image/20190919_095020_ssc10_u0001_pansharpened.tif'
    """Canada"""
    # shape_in = r'/home/skm/SKM/WORK/USA/Queensland Mosaics/img_ori/Capricorn_Wide_Bay_2017_20cm_Mosaic_Result/building/v1_Capricorn_Wide_Bay_2017_20cm_Mosaic_resize03_v382_4326.shp'
    # shp_out = r'/home/skm/SKM/WORK/USA/Queensland Mosaics/img_ori/Capricorn_Wide_Bay_2017_20cm_Mosaic_Result/building/rectify/v1_Capricorn_Wide_Bay_2017_20cm_Mosaic_resize03_v382_4326.shp'
    # img_url = r'/home/skm/SKM/WORK/USA/Queensland Mosaics/img_ori/Capricorn_Wide_Bay_2017_20cm_Mosaic_Result/Capricorn_Wide_Bay_2017_20cm_Mosaic_resize03.tif'
    # create_mask(shape_in, img_url, shp_out, scale=1)

    """Chay thu"""
    # import time
    # x = time.time()
    # shape_in = r'/home/skm/SKM/Machine_learning_full_model/temp/baf29f1b0d5d413489c53b405c45cc59/out_path.shp'
    # shp_out = r'/home/skm/SKM/Machine_learning_full_model/temp/baf29f1b0d5d413489c53b405c45cc59/out_path_rec.shp'
    # img_url = r'/home/skm/SKM/Machine_learning_full_model/temp/baf29f1b0d5d413489c53b405c45cc59/Nunukan_3B.tif'
    # create_mask(shape_in, img_url, shp_out, scale=1)
    # print((time.time()-x)/60)


    """For cho ca folder"""
    import os, glob
    def get_list_name_file(path_folder, name_file = '*.tif'):
        list_img_dir = []
        for file_ in glob.glob(os.path.join(path_folder, name_file)):
            _, tail = os.path.split(file_)
            list_img_dir.append(tail)
        return list_img_dir

    import time
    x = time.time()
    # folder_shp = r"/home/skm/SKM/WORK/Buiding/UAE_building/building/img/predict_model__building_usa_resize_03_51220220214T2203"
    # fodler_img = r"/home/skm/SKM/WORK/Buiding/UAE_building/building/img"
    # foder_out = r"/home/skm/SKM/WORK/Buiding/UAE_building/building/img/predict_model__building_usa_resize_03_51220220214T2203/rectify"

    folder_shp = r"/home/skm/SKM_OLD/public/MaxarARD/building_3band_maxar_img_03m_and_05m20220309T1607"
    fodler_img = r"/home/skm/SKM_OLD/public/MaxarARD"
    foder_out = r"/home/skm/SKM_OLD/public/MaxarARD/building_3band_maxar_img_03m_and_05m20220309T1607_rectify"

    list_name = get_list_name_file(fodler_img, name_file = '*.tif')
    for name in list_name:
        shape_in = os.path.join(folder_shp, name[:-4] + '.shp')
        shp_out = os.path.join(foder_out, name[:-4] + '.shp')
        img_url = os.path.join(fodler_img, name)
        create_mask(shape_in, img_url, shp_out, scale=1)
    print((time.time()-x)/60)    