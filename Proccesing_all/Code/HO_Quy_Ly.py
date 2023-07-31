import gdal
import numpy as np
import geojson
from osgeo import gdal, ogr, osr
from pyproj import Proj, transform
import cv2

def point_to_geopoint(point, geotransform):
    topleftX = geotransform[0]
    topleftY = geotransform[3]
    XRes = geotransform[1]
    YRes = geotransform[5]
    geopoint = (topleftX + point[0] * XRes, topleftY + point[1] * YRes)
    return geopoint

def polygon_to_geopolygon(polygon, geotransform):
    temp_geopolygon = []
    for point in polygon:
        geopoint = point_to_geopoint(point, geotransform)
        temp_geopolygon.append(geopoint)
    geopolygon = tuple(temp_geopolygon)
    return geopolygon

def list_polygon_to_list_geopolygon(list_polygon, geotransform):
    list_geopolygon = []
    for polygon in list_polygon:
        geopolygon = polygon_to_geopolygon(polygon, geotransform)
        list_geopolygon.append(geopolygon)
    # p_geocal = Pool(processes=core_of_computer)
    # result = p_geocal.map(partial(polygon_to_geopolygon,geotransform=geotransform), list_polygon)
    # p_geocal.close()
    # p_geocal.join()
    # list_geopolygon = result
    return list_geopolygon

def exportResult2(list_polygon, geotransform, projection, outputFileName, driverName):
    list_geopolygon = list_polygon_to_list_geopolygon(list_polygon, geotransform)
#    list_geopolygon = transformToLatLong(list_geopolygon, projectionString)
    # print(list_geopolygon)
    driver = ogr.GetDriverByName(driverName)
    data_source = driver.CreateDataSource(outputFileName)
#    projection = osr.SpatialReference()
#    projection.ImportFromProj4(projectionString)
    outLayer = data_source.CreateLayer("Building Footprint", projection, ogr.wkbPolygon)
    featureDefn = outLayer.GetLayerDefn()
    for geopolygon in list_geopolygon:
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for point in geopolygon:
            ring.AddPoint(point[0], point[1])
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(polygon)
        outLayer.CreateFeature(outFeature)
    ###############################################################################
    # destroy the feature
    outFeature = None
    # destroy the feature
    outLayer = None
    # Close DataSources
    data_source = None


def contour_to_polygon(contour):
    list_point = []
    try:
        for point in contour:
            [x,y] = point[0]
            point_in_polygon = (x,y)
            list_point.append(point_in_polygon)
        # print(contour[0])
        [x,y] = contour[0][0]
        point_in_polygon = (x,y)
        list_point.append(point_in_polygon)
        poly = tuple(list_point)
    except Exception:
        poly = ()
    return poly


def list_contour_to_list_polygon(list_contour):
    list_polygon = []
    for contour in list_contour:
        poly = contour_to_polygon(contour)
        list_polygon.append(poly)
    return list_polygon




def main(img_path, outputFileName1, outputFileName2):

    
    dataset = gdal.Open(img_path)
    numpy_tif = dataset.ReadAsArray()
    geotransform = dataset.GetGeoTransform()
    projection = osr.SpatialReference(dataset.GetProjectionRef())
    driverName = "GeoJSON"

    print(numpy_tif[0],len(numpy_tif[0]))

    im1, contours1, hierarchy = cv2.findContours(numpy_tif[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im2, contours2, hierarchy = cv2.findContours(numpy_tif[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    list_poly_1 = list_contour_to_list_polygon(contours1)
    list_poly_2 = list_contour_to_list_polygon(contours2)

    exportResult2(list_poly_1, geotransform, projection, outputFileName1, driverName)
    exportResult2(list_poly_2, geotransform, projection, outputFileName2, driverName)

if __name__ == '__main__':
    img_path = r"/media/khoi/Image/ThaiLand/sugarcane/aaaaaaa.tif"
    outputFileName1 = r"/media/khoi/Image/ThaiLand/sugarcane/nhanh/aaaaaaa1.json"
    outputFileName2 = r"/media/khoi/Image/ThaiLand/sugarcane/nhanh/bbbbbbb1.json"
    main(img_path, outputFileName1, outputFileName2)