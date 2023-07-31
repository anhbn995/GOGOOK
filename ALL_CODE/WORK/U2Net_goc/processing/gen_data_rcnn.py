import glob
import rasterio
import numpy as np
import glob, os
from itertools import product
import rasterio
from rasterio import windows
from shapely.geometry import Polygon, box, mapping
from shapely.strtree import STRtree
import datetime
import geopandas
import copy
import math
import json
from tqdm import tqdm

def get_tiles(ds, width, height):
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, 400), range(0, nrows, 400))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    offset = []
    for col_off, row_off in offsets:
        if row_off + _width > nrows:
            row_off = nrows - _width
        if  col_off + _height > nols:
            col_off = nols - _height
        offset.append((col_off, row_off))
    offset = set(offset)
    print(len(offset))
    for col_off, row_off in tqdm(offset): 
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform

def hoangnam(_images, _annotations, _image_id, _annotation_id, types):
    check = False 
    for o in b.query(boxx):  
        if o.intersects(boxx):
            check = True
            _annotation_id = _annotation_id + 1
            ggg = o.intersection(boxx)
            _segmentation = []
            for a in ggg.exterior.coords:
                x = float(int((a[0] - transform[2])/ transform[0]))
                y = float(int((a[1] - transform[5])/ transform[4]))
                _segmentation.extend((x,y))
            _area = round((ggg.area/transform[0]/-transform[4]),2)
            _bbox = [float(int((ggg.bounds[3] - transform[5])/ transform[4])),
                    float(int((ggg.bounds[0] - transform[2])/ transform[0])),
                    float(math.ceil((ggg.bounds[1] - ggg.bounds[3])/ transform[4])),
                    float(math.ceil((ggg.bounds[2] - ggg.bounds[0])/ transform[0]))]
            annotation_info = {
                "id": _annotation_id,
                "image_id": _image_id,
                # "category_id": my_geoseries.loc[my_geoseries.geometry == o,'label'].iloc[0].astype('int8'),
                "category_id": 1,
                "iscrowd": 0,
                "area": _area,
                "bbox": _bbox,
                "segmentation": [_segmentation],
                "width": _width,
                "height": _height,
            }
            _annotations.append(annotation_info)

    if check: 
        _file_name = output_box.format('{0:003d}'.format(_image_id))
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
        outpath_image = os.path.join(file, types, 'images/',output_box.format('{0:003d}'.format(_image_id)))
        with rasterio.open(outpath_image, "w", **meta, compress='lzw') as dest:
            dest.write(inds.read(window=window))
        _image_id+=1
        
    return _images, _annotations, _image_id, _annotation_id

_final_object_train = {}
_final_object_train["info"]= {
                        "contributor": "crowdAI.org",
                        "about": "Dataset for crowdAI Mapping Challenge",
                        "date_created": datetime.datetime.utcnow().isoformat(' '),
                        "description": "crowdAI mapping-challenge dataset",
                        "url": "https://www.crowdai.org/challenges/mapping-challenge",
                        "version": "1.0",
                        "year": 2018
                        }

_final_object_train["categories"]=[
                {
                    "id": 1,
                    "name": "ship",
                    "supercategory": "ship"
                }
            ]

_final_object_val = copy.copy(_final_object_train)

date_captured=datetime.datetime.utcnow().isoformat(' ')
license_id=1
coco_url=""
flickr_url=""

_images_train = []
_annotations_train = []
_image_id_train = 1
_annotation_id_train = 0

_images_val = []
_annotations_val = []
_image_id_val = 1
_annotation_id_val = 0

output_box = 'box_{}.tif'
file = '/mnt/Nam/official_model/Mask_RCNN/data/'
img_path = '/mnt/Nam/public/car_detection/Mt_Isa_2017_10cm_Mosaic.tif'
shp_path = '/mnt/Nam/public/car_detection/car_detection/label_car.shp'
# image_list = ['box_6.tif','box_7.tif','box_8.tif','box_9.tif','box_10.tif']

if not os.path.exists(file+'train'):
    os.makedirs(file+'train/images/')
    
if not os.path.exists(file+'val'):
    os.makedirs(file+'val/images/')

with rasterio.open(img_path) as inds:
    _width, _height = 512, 512
    meta = inds.meta.copy()
    projstr = inds.crs.to_string()
        
    my_geoseries = geopandas.read_file(shp_path)
    my_geoseries = my_geoseries.to_crs(projstr)
    b= STRtree(my_geoseries['geometry'])

    for window, transform in get_tiles(inds, _width, _height):
        meta['transform'] = transform
        meta['width'], meta['height'] = window.width, window.height
        min = transform * (0, 0)
        max = transform * (_width, _height)
        boxx = box(*np.minimum(min,max), *np.maximum(min,max))
        if np.random.random_sample()>0.2499:
            _images_train, _annotations_train, _image_id_train, _annotation_id_train = hoangnam(_images_train, _annotations_train, _image_id_train, _annotation_id_train, 'train')
        else:
            _images_val, _annotations_val, _image_id_val, _annotation_id_val = hoangnam(_images_val, _annotations_val, _image_id_val, _annotation_id_val, 'val')
            

_final_object_train["images"]=_images_train
_final_object_train["annotations"]=_annotations_train 
_final_object_val["images"]=_images_val
_final_object_val["annotations"]=_annotations_val 
print('2')
fp = open(file + 'train/annotation.json', "w+")
fp.write(json.dumps(_final_object_train))
fp = open(file + 'val/annotation.json', "w+")
fp.write(json.dumps(_final_object_val))