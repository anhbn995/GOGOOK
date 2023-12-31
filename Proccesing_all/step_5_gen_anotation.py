import tqdm
import glob, os
import json
import sys
from osgeo import gdal, gdalconst, ogr, osr
import shapefile as shp
import numpy as np
from pyproj import Proj, transform
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
import datetime
import cv2
from tqdm import *
image_dir = os.path.abspath(sys.argv[1])
parent = os.path.dirname(image_dir)
foder_name = os.path.basename(image_dir)
shape_dir = os.path.abspath(sys.argv[2])
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
def create_list_annotation(image_name):
    "đọc shapefile"
    shape_path = os.path.join(shape_dir,image_name+'.shp')
    sf=shp.Reader(shape_path)
    my_shapes=sf.shapes()
    my_shapes_list = list(map(lambda shape: shape.points, my_shapes))
    # get epsg
    driverName = "ESRI Shapefile"
    driver = ogr.GetDriverByName(driverName)
    dataSource = driver.Open(shape_path, 0)
    layer = dataSource.GetLayer()
    crs = layer.GetSpatialRef()
    # epsr1 =  crs.GetAttrValue('AUTHORITY',1)
    "đọc ảnh"
    driver = gdal.GetDriverByName('GTiff')
    filename = os.path.join(image_dir,image_name+'.tif')
    # print(filename)
    dataset = gdal.Open(filename)
    proj = osr.SpatialReference(wkt=dataset.GetProjection())
    # print(crs)
    # epsr2 = (proj.GetAttrValue('AUTHORITY',1))
    "chuyen toa do "
    epsr2 = 4326
    epsr1 = 4326
    #epsr1 = 26911
    # print(epsr1,epsr2)
    # '+proj=merc +lon_0=0 +lat_ts=0 +x_0=0 +y_0=0 +a=6378137 +b=6378137 +units=m +no_defs'
    # inProj = Proj('+proj=merc +lon_0=0 +lat_ts=0 +x_0=0 +y_0=0 +a=6378137 +b=6378137 +units=m +no_defs')
    inProj = Proj(init='epsg:%s'%(epsr1))
    outProj = Proj(init='epsg:%s'%(epsr2))

    list_list_point_convert = []
    for shapes in my_shapes_list:
        list_point=[]
        for point in shapes:
            longs,lat = point[0],point[1]
            x,y = transform(inProj,outProj,longs,lat)
            list_point.append((x,y))
        list_list_point_convert.append(list_point)

    # data = dataset.ReadAsArray()
    # img = np.array(data).swapaxes(0,1).swapaxes(1,2)
    "chuyen sang toa do pixel"
    transformmer = dataset.GetGeoTransform()

    xOrigin = transformmer[0]
    yOrigin = transformmer[3]
    pixelWidth = transformmer[1]
    pixelHeight = -transformmer[5]

    list_list_point=[]
    for points_list in list_list_point_convert :
    #    i=i+1
        lis_poly=[]
        for point in points_list:
            col = int((point[0] - xOrigin) / pixelWidth)
            row = int((yOrigin - point[1] ) / pixelHeight)
            lis_poly.append([col,row])
        lis_poly = np.asarray(lis_poly,dtype = np.int)
        list_list_point.append(lis_poly)
    list_cnt = list_list_array_to_list_contour(list_list_point)
    width, height = dataset.RasterXSize, dataset.RasterYSize
    return height, width, list_cnt
def main():
    _final_object = {}
    _final_object["info"]= {
                            "contributor": "crowdAI.org",
                            "about": "Dataset for crowdAI Mapping Challenge",
                            "date_created": datetime.datetime.utcnow().isoformat(' '),
                            "description": "crowdAI mapping-challenge dataset",
                            "url": "https://www.crowdai.org/challenges/mapping-challenge",
                            "version": "1.0",
                            "year": 2018
                            }
                        
    _final_object["categories"]=[
                    {
                        "id": 100,
                        "name": "building",
                        "supercategory": "building"
                    }
                ]
    date_captured=datetime.datetime.utcnow().isoformat(' ')
    license_id=1
    coco_url=""
    flickr_url=""
    _images = []
    _annotations = []
    _list_image = create_list_id(image_dir)
    _image_id = 0
    _annotation_id = 0
    with tqdm(total=len(_list_image)) as pbar:
        for _image_name in _list_image:
            pbar.update()
            _image_id = _image_id + 1
            _file_name = _image_name + '.tif'
            _height, _width, list_annotation = create_list_annotation(_image_name)
            image_info = {
                        "id": _image_id,
                        "file_name": _file_name,
                        "width": _width,
                        "height": _height,
                        "date_captured": date_captured,
                        "license": license_id,
                        "coco_url": coco_url,
                        "flickr_url": flickr_url
                        }
            _images.append(image_info)
            for annotation in list_annotation:
                _annotation_id = _annotation_id + 1
                _area = cv2.contourArea(annotation)
                x,y,w,h = cv2.boundingRect(annotation)
                _bbox = [y,x,h,w]
                _segmentation = [list(annotation.astype(np.float64).reshape(-1))]
                annotation_info = {
                    "id": _annotation_id,
                    "image_id": _image_id,
                    "category_id": 100,
                    "iscrowd": 0,
                    "area": _area,
                    "bbox": _bbox,
                    "segmentation": _segmentation,
                    "width": _width,
                    "height": _height,
                }
                _annotations.append(annotation_info)
    _final_object["images"]=_images
    _final_object["annotations"]=_annotations
    # print(_final_object)
    fp = open(os.path.join(parent,'annotation_step2.json'), "w")
    print("Writing JSON...")
    fp.write(json.dumps(_final_object))

if __name__ == "__main__":
    main()

#folder imag, folder shap da chia nho coi nhu 2 thu muc trong tmp

