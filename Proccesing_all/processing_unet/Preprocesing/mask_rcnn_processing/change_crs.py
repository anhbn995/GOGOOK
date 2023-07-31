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
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from osgeo import ogr
import shapely.wkt
from shapely.geometry.multipolygon import MultiPolygon
from tqdm import *
foder_shape = os.path.abspath(sys.argv[1])
foder_name = os.path.basename(foder_shape)
out_epsg = int(sys.argv[2])
parent = os.path.dirname(foder_shape)
core = multiprocessing.cpu_count()
import shapefile as shp
def load_shapefile(id_shape,path_change_crop,out_epsg):
    # union_poly = ogr.Geometry(ogr.wkbPolygon)
    shape_path = os.path.join(foder_shape,id_shape+'.shp')
    sf=shp.Reader(shape_path)
    my_shapes=sf.shapes()
    my_shapes_list = list(map(lambda shape: shape.points, my_shapes))
    # print(my_shapes_list)
    # union_poly = ogr.Geometry(ogr.wkbMultiPolygon)

    # make the union of polygons
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(shape_path, 0)
    layer_in = dataSource.GetLayer()
    srs = layer_in.GetSpatialRef()
    geom_type = layer_in.GetGeomType()
    epsr1 =  srs.GetAttrValue('AUTHORITY',1)
    geom_type = layer_in.GetGeomType()
    layer_defn = layer_in.GetLayerDefn()
    list_polygon = []
    for feature in layer_in:
        geom =feature.GetGeometryRef()
        if geom is not None:
            if (geom.GetGeometryName()) == "POLYGON":
                # print(True)
    #         union_poly.AddGeometry(geom)
    #     # print(geom) 
    # result = union_poly.UnionCascaded()
    # print(result)
                shape_obj = (shapely.wkt.loads(geom.ExportToWkt()))
                polygon = (tuple(shape_obj.exterior.coords))
                # geo2d = ogr.Geometry(ogr.wkbPolygon)
                # geo2d.AddGeometry(polygon)
                # list_polygon.append(polygon)
                # poly_rs = []
                # ring = ogr.Geometry(ogr.wkbLinearRing)
                # for point in polygon:
                #     ring.AddPoint(point[0],point[1])
                # poly = ogr.Geometry(ogr.wkbPolygon)
                # print(ring)
                # poly.AddGeometry(ring)
                # union_poly.AddGeometry(poly) 
                poly_rs = []
                for point in polygon:
                    poly_rs.append((point[0],point[1]))
                poly_rs = tuple(poly_rs) 
                if len(poly_rs)>1:  
                    list_polygon.append(Polygon(poly_rs))
            elif (geom.GetGeometryName()) == "MULTIPOLYGON":
                for geo1 in geom:
                    if geo1 is not None:
                        shape_obj = (shapely.wkt.loads(geo1.ExportToWkt()))
                        polygon1 = (tuple(shape_obj.exterior.coords))
                        # geo2d = ogr.Geometry(ogr.wkbPolygon)
                        # geo2d.AddGeometry(polygon)
                        # list_polygon.append(polygon)
                        # poly_rs = []
                        # ring = ogr.Geometry(ogr.wkbLinearRing)
                        # for point in polygon1:
                        #     ring.AddPoint(point[0],point[1])
                        #     # print(point[0],point[1])
                        # # print(ring)
                        # poly = ogr.Geometry(ogr.wkbPolygon)
                        # poly.AddGeometry(ring)
                        # union_poly.AddGeometry(poly)    
                        poly_rs = []
                        for point in polygon1:
                            poly_rs.append((point[0],point[1]))
                        poly_rs = tuple(poly_rs)
                        if len(poly_rs)>1:
                            list_polygon.append(Polygon(poly_rs))
    b = [aa for aa in list_polygon if aa.geom_type == 'Polygon' ]
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds_out = driver.CreateDataSource(os.path.join(path_change_crop,id_shape+'.shp'))
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(out_epsg)
    layer_out = ds_out.CreateLayer("Building area", srs = outSpatialRef, geom_type = ogr.wkbPolygon)
    layer_defn = layer_out.GetLayerDefn()
    for polygon in b:
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for point in polygon.exterior.coords:
            # print(point)
            ring.AddPoint(point[0], point[1])
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        outFeature = ogr.Feature(layer_defn)
        outFeature.SetGeometry(poly)
        layer_out.CreateFeature(outFeature)
    outFeature = None
    # destroy the feature
    outLayer = None
    # Close DataSources
    data_source = None
    return True
def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.shp"):
        if file[0:6]=='Amreli':
            list_id.append(file[:-4])
        # print(file[:-4])
    return list_id

def main_change():
    list_id = create_list_id(foder_shape)
    # print(list_id)
    if not os.path.exists(os.path.join(parent,foder_name+'_change')):
        os.makedirs(os.path.join(parent,foder_name+'_change'))
    path_change_crop = os.path.join(parent,foder_name+'_change')
    p_change_crs = Pool(processes=core)
    pool_result = p_change_crs.imap_unordered(partial(load_shapefile,path_change_crop=path_change_crop,out_epsg=out_epsg), list_id)
    with tqdm(total=len(list_id)) as pbar:
        for i,_ in tqdm(enumerate(pool_result)):
            pbar.update()
    p_change_crs.close()
    p_change_crs.join()
if __name__ == "__main__":
    x1 = time.time()
    main_change()
    print(time.time() - x1, "second")

