from osgeo import osr
import numpy as np
from osgeo import gdal, ogr, osr
import cv2
import pandas as pd
import rasterio.features
import shapely.ops
import shapely.geometry
import shapely.wkt
import shapefile
from pyproj import Proj, transform
import os
from globalmaptiles import GlobalMercator

def get_poly(mask, min_polygon_thres = 30):
    df = mask_to_poly(mask, min_polygon_thres)
    list_polygon = []
    for poly in df["poly"]:
        polygon = (tuple(poly.exterior.coords))
        list_polygon.append(polygon)
    del df
    return list_polygon

def mask_to_poly(mask, min_polygon_area_th=30):
    shapes = rasterio.features.shapes(mask.astype(np.int16), mask > 0)
    mp = shapely.ops.cascaded_union(
        shapely.geometry.MultiPolygon([
            shapely.geometry.shape(shape)
            for shape, value in shapes
        ]))

    if isinstance(mp, shapely.geometry.Polygon):
        df = pd.DataFrame({
            'area_size': [mp.area],
            'poly': [mp],
        })
    else:
        df = pd.DataFrame({
            'area_size': [p.area for p in mp],
            'poly': [p for p in mp],
        })
    df = df[df.area_size > min_polygon_area_th]
    return df

def save_poly_to_shape_file(list_polygon, img_path, shape_file_path):
    raster_data = gdal.Open(img_path)
    driverName = "ESRI Shapefile"
    outputFileName = shape_file_path
    geotransform = raster_data.GetGeoTransform()
    projection = osr.SpatialReference(raster_data.GetProjectionRef())
    exportShapeFile(list_polygon, geotransform, projection, outputFileName, driverName)
    del raster_data
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

def exportShapeFile(list_polygon, geotransform, projection, outputFileName, driverName):
    list_geopolygon = list_polygon_to_list_geopolygon(
        list_polygon, geotransform)
    driver = ogr.GetDriverByName(driverName)
    data_source = driver.CreateDataSource(outputFileName)
    outLayer = data_source.CreateLayer("ChangeDetection", projection, ogr.wkbPolygon)
    featureDefn = outLayer.GetLayerDefn()
    epsg = 'epsg:' + str(projection.GetAttrValue('AUTHORITY',1))
    for i in range(len(list_geopolygon)):
        geopolygon = list_geopolygon[i]
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for point in geopolygon:
            ring.AddPoint(point[0], point[1])
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(polygon)
        outLayer.CreateFeature(outFeature)
    outFeature = None
    outLayer = None
    data_source = None
    return True

def point_to_geopoint(point, geotransform):
    topleftX = geotransform[0]
    topleftY = geotransform[3]
    XRes = geotransform[1]
    YRes = geotransform[5]
    geopoint = (topleftX + point[0] * XRes, topleftY + point[1] * YRes)
    return geopoint

def mask_to_shp(image_name, imageID):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 1000000000
    # im = Image.open(image_name)
    # width, height = im.size
    # del im

    img=gdal.Open(image_name)
    inputArray=img.ReadAsArray()

    # driver = gdal.GetDriverByName('GTiff')
    # dst_ds = driver.Create('output.tif',xsize=width, ysize=height, bands=1, eType=gdal.GDT_Byte)

    # ls = os.listdir('./uploaded/{}/17'.format(imageID))
    # arrx = []
    # arry = []
    # for dir in ls:
    #     lf = os.listdir('./uploaded/{}/17/{}'.format(imageID, dir))
    #     arrx.append(int(dir))
    #     for fl in lf:
    #         a, b = fl.split(".")
    #         arry.append(int(a))
    # mercator = GlobalMercator()

    # p1 = mercator.TileLatLonBounds(max(arrx),max(arry),17)
    # p2 = mercator.TileLatLonBounds(min(arrx),min(arry),17)

    # dst_ds.SetGeoTransform([p2[1], (p1[3]-p2[1])/float(width), 0, p1[2], 0, (p2[0]-p1[2])/float(height)])

    # srs = osr.SpatialReference()
    # srs.ImportFromEPSG(4326)
    # dst_ds.SetProjection(srs.ExportToWkt())

    # for i in range(1):
    #     dst_ds.GetRasterBand(i+1).WriteArray(inputArray)

    # del dst_ds

    list_poly = get_poly(inputArray)
    save_poly_to_shape_file(list_poly, image_name, 'output.shp')
    return 'output.shp'

def shp_to_geojson(name_shp, imageID):
    geoName = imageID + ".geojson"
    geofile = "./uploaded/" + imageID + "/" + imageID + ".geojson"
    # inProj = Proj(init='epsg:4326')
    # outProj = Proj(init='epsg:4326')
    reader = shapefile.Reader(name_shp)
    fields = reader.fields[1:]
    field_names = [field[0] for field in fields]
    buffer = []
    for sr in reader.shapeRecords():
        atr = dict(zip(field_names, sr.record))
        geom = sr.shape.__geo_interface__
        arr = ()
        for cor in geom['coordinates'][0]:
            # x1, y1 = cor
            # cor = transform(inProj,outProj,x1,y1)
            arr += (cor,)
        geom['coordinates'] = (arr,)
        buffer.append(dict(type="Feature", geometry=geom, properties=atr))
    
    # write the GeoJSON file
    from json import dumps
    geojson = open(geofile, "w")
    geojson.write(dumps({"id": imageID, "type": "FeatureCollection",\
        "features": buffer}, indent=2) + "\n")
    geojson.close()
    return geoName

if __name__ == "__main__":
    mask_to_shp('Bhuvan_RGBN_Geo_3_C43E13.tif', 'lul')