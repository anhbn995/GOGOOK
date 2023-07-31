import numpy as np # linear algebra
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
import pandas as pd
import csv
from matplotlib import pyplot as plt
image_dir = os.path.abspath(sys.argv[1])
parent = os.path.dirname(image_dir)
foder_name = os.path.basename(image_dir)
csv_path = os.path.abspath(sys.argv[2])

split = float(sys.argv[3])
def read_data_csv(csv_path):
    file=open(csv_path, "r")  
    reader = csv.reader(file)  
    data = [line for line in reader]
    return data

def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.jpg"):
        list_id.append(file)
        # print(file[:-4])
    return list_id

def rleToMask(rleString,height,width):
  rows,cols = height,width
  try:
    rleNumbers = [(int(float(numstring))) for numstring in rleString.split(' ')]
    rlePairs = np.array(rleNumbers).reshape(-1,2)
    img = np.zeros(rows*cols,dtype=np.uint8)
    for index,length in rlePairs:
        index -= 1
        img[index:index+length] = 255
    img = img.reshape(cols,rows)
    img = img.T
  except Exception:
      img = np.zeros((height,width), dtype= np.uint8)
  return img

def create_list_annotation(image_name,data):
    list_rle = [line[1] for line in data if line[0] == image_name]
    filename = os.path.join(image_dir,image_name)
    dataset = gdal.Open(filename)
    width, height = dataset.RasterXSize, dataset.RasterYSize
    list_anotation = []
    for rle in list_rle:
        mask = rleToMask(rle,height,width)
        mask1 = mask.copy().astype(np.uint8)
        # print(mask)
        # plt.imshow(mask)
        # plt.show()
        im2, contours, hierarchy = cv2.findContours(mask1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        list_anotation.extend([cnt for cnt in contours if cv2.contourArea(cnt) > 1.0 ])
    return height, width, list_anotation

def main():
    data = read_data_csv(csv_path)
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
                        "id": 101,
                        "name": "ship",
                        "supercategory": "ship"
                    }
                ]
    date_captured=datetime.datetime.utcnow().isoformat(' ')
    license_id=1
    coco_url=""
    flickr_url=""
    _images = []
    _annotations = []
    _list_image = create_list_id(image_dir)
    np.random.shuffle(_list_image)
    count = len(_list_image)
    cut_idx = int(round(count*split))
    print(cut_idx)
    # train_list = _list_image[0:cut_idx]

    # val_list = _list_image[cut_idx:count]
    # val_list = [id_image for id_image in _list_image if id_image not in train_list]
    # _list_image = _list_image[100000:len(_list_image)]
    _list_image = _list_image[0:cut_idx]
    _image_id = 0
    _annotation_id = 0
    with tqdm(total=len(_list_image)) as pbar:
        for _image_name in _list_image:
            pbar.update()
            _image_id = _image_id + 1
            _file_name = _image_name
            _height, _width, list_annotation = create_list_annotation(_image_name,data)
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
                    "category_id": 101,
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
    fp = open(os.path.join(parent,'annotation.json'), "w")
    print("Writing JSON...")
    fp.write(json.dumps(_final_object))

if __name__ == "__main__":
    main()