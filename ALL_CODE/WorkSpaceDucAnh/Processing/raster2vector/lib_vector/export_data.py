import geojson
from osgeo import gdal, ogr, osr
from pyproj import Proj, transform
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
core_of_computer = multiprocessing.cpu_count()
def export_geojson_str(geo_polygons, geo_transform, projection_str):
    features = []
    list_geopolygon = list_polygon_to_list_geopolygon(geo_polygons, geo_transform)
    list_geopolygon = transformToLatLong(list_geopolygon, projection_str)
    for geo_polygon in list_geopolygon:
        # geo_polygon = np.array(geo_polygon).tolist()
        polygon = geojson.Polygon([geo_polygon])
        feature = geojson.Feature(geometry=polygon)
        features.append(feature)
    return geojson.dumps(geojson.FeatureCollection(features))

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

def transformToLatLong(list_geopolygon, projectionString):
    projectionString = projectionString
    latlongProjection = osr.SpatialReference()
    #    latlongProjection = Proj(init='epsg:4326')#not run
    latlongProjection.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs ')
    #    latlongProjection.ImportFromProj4(init='epsg:4326')#not run
    projection = osr.SpatialReference()
    projection.ImportFromProj4(projectionString)
    if projection.IsSame(latlongProjection):
        print('Same')
        return list_geopolygon
    else:
        print('Different')
        new_list_geopolygon = []
        projectionOld = Proj(projectionString)
        #        projectionNew = Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs ')#run ok
        projectionNew = Proj(init='epsg:4326')

        for geopolygon in list_geopolygon:
            new_geopolygon = []
            for point in geopolygon:
                newPoint = transform(projectionOld, projectionNew, point[0], point[1])
                new_geopolygon.append(newPoint)
            new_list_geopolygon.append(new_geopolygon)
        return new_list_geopolygon


def exportResult(list_polygon, geotransform, projectionString, outputFileName, driverName):
    list_geopolygon = list_polygon_to_list_geopolygon(list_polygon, geotransform)
    list_geopolygon = transformToLatLong(list_geopolygon, projectionString)
    print(list_geopolygon)
    driver = ogr.GetDriverByName(driverName)
    data_source = driver.CreateDataSource(outputFileName)
    projection = osr.SpatialReference()
    projection.ImportFromProj4(projectionString)
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