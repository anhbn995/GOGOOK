import json
from shapely.geometry import Polygon
from utils.globalmaptiles import GlobalMercator
import requests

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO, BytesIO

from skimage import io
import numpy as np
import random
import os
import cv2
from tqdm import *
from utils import bing
from retrying import retry
import concurrent.futures
import time
googleMapURLs = [
                # "http://khms.google.com/kh/v=908?x={x}&y={y}&z={z}"
                # 'https://khms0.google.com/kh/v=908?x={x}&y={y}&z={z}'
                # 'https://khms1.google.com/kh/v=908?x={x}&y={y}&z={z}',
                # 'https://khms3.google.com/kh/v=908?x={x}&y={y}&z={z}',
                # 'https://khms2.google.com/kh/v=908?x={x}&y={y}&z={z}',
                "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
                # "https://mt2.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
                 ]

# bingMapURLs = [
#     'http://ecn.t0.tiles.virtualearth.net/tiles/a{qKey}.jpeg?g=2',
#     'http://a0.ortho.tiles.virtualearth.net/tiles/a{qKey}.jpeg?g=2',
#     "http://ecn.t3.tiles.virtualearth.net/tiles/a{qKey}.jpeg?g=2"
# ] 
bingMapURLs = [
    # "http://185.52.192.133/server0/tiles/a{qKey}.jpeg?g=2",
# "http://185.52.192.133/server1/tiles/a{qKey}.jpeg?g=2",
# "http://185.52.192.133/server2/tiles/a{qKey}.jpeg?g=2",
# "http://185.52.192.133/server3/tiles/a{qKey}.jpeg?g=2",
# "http://185.52.194.187/server0/tiles/a{qKey}.jpeg?g=2",
# "http://185.52.194.187/server1/tiles/a{qKey}.jpeg?g=2",
# "http://185.52.194.187/server2/tiles/a{qKey}.jpeg?g=2",
# "http://185.52.194.187/server3/tiles/a{qKey}.jpeg?g=2",
# "http://185.52.192.82/server0/tiles/a{qKey}.jpeg?g=2",
# "http://185.52.192.82/server1/tiles/a{qKey}.jpeg?g=2",
# "http://185.52.192.82/server2/tiles/a{qKey}.jpeg?g=2",
# "http://185.52.192.82/server3/tiles/a{qKey}.jpeg?g=2",
'http://ecn.t0.tiles.virtualearth.net/tiles/a{qKey}.jpeg?g=2',
'http://a0.ortho.tiles.virtualearth.net/tiles/a{qKey}.jpeg?g=2',
"http://ecn.t3.tiles.virtualearth.net/tiles/a{qKey}.jpeg?g=2"
]


esriMapURLs = [
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ]
def getExtentOfPolygon(points):
    geometry = Polygon(points)
    return geometry.bounds

def getGridMatrix(extent, zoomLevel):
    lon, lat, lonmax, latmax = extent
    mercator = GlobalMercator()
    tz = zoomLevel
    minx, miny = mercator.LatLonToMeters(lat, lon)
    tminx, tminy = mercator.MetersToTile(minx, miny, tz)
    maxx, maxy = mercator.LatLonToMeters(latmax, lonmax)
    tmaxx, tmaxy = mercator.MetersToTile(maxx, maxy, tz)
    return(tminx,tminy,tmaxx,tmaxy)

def getGridBound(xyz):
    tx,ty,tz = xyz
    mercator = GlobalMercator()
    latLonBound = mercator.TileLatLonBounds(tx, ty, tz)
    return (latLonBound[1], latLonBound[0], latLonBound[3], latLonBound[2])

def latlongToLongLat(points):
    return map(lambda x:  x[::-1], points)

def intersect(bound, polygon):
    # (xmin, ymin, xmax, ymax) -> topleft, topright, bottomright, bottomleft
    polybound = bound_to_polygon(bound)
    return polybound.intersects(polygon)

def bound_to_polygon(bound):
    tl = (bound[0],bound[1])
    tr = (bound[2],bound[1])
    br = (bound[2],bound[3])
    bl = (bound[0],bound[3])
    polybound = Polygon([tl,tr,br,bl,tl])
    return polybound

def tileToGoogleTile(xyz):
    tx, ty, tz = xyz
    mercator = GlobalMercator()
    gx, gy = mercator.GoogleTile(tx, ty, tz)
    return (gx, gy, tz)

@retry(
    wait_fixed=1000
)
def download_tile_bing(url,save_path):
    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if os.path.isfile(save_path):
        pass
    else:
        headers = {}
        tile = requests.get(url, headers=headers)
        result =  io.imread(BytesIO(tile.content))
        cv2.imwrite(save_path,cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return True

def download_bing_tile(data,download_path):
    mercator = GlobalMercator()
    list_url = []
    list_save_path = []
    for i in range(len(data)):
        url_index = random.randint(0, len(bingMapURLs)-1)
        item = data[i]
        tx, ty, tz = item[1]["x"],item[1]["y"],item[1]["z"]
        qKey = mercator.QuadTree(tx, ty, tz)
        url = bingMapURLs[url_index].format(**{'qKey':qKey})
        save_path = os.path.join(download_path,str(tz),str(tx),"{}.png".format(str(ty)))
        # print(i,len(data),i/len(data))
        list_url.append(url)
        list_save_path.append(save_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
        results = list(tqdm(executor.map(download_tile_bing, list_url,list_save_path), total=len(list_url)))
        download_tile_esri(url,save_path)
    download_tile_bing(url,save_path)
            

@retry(
    wait_fixed=1000
)
def download_tile_esri(url,save_path):
    time.sleep(0.2)
    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if os.path.isfile(save_path):
        pass
    else:
        headers = {}
        tile = requests.get(url, headers=headers)
        result =  io.imread(BytesIO(tile.content))
        cv2.imwrite(save_path,cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    return True

def download_esri_tile(data,download_path):
    mercator = GlobalMercator()
    list_url = []
    list_save_path = []
    for i in range(len(data)):
        url_index = random.randint(0, len(googleMapURLs)-1)
        item = data[i]
        tx2, ty2, tz2 = item[2]["x"],item[2]["y"],item[2]["z"]
        tx, ty, tz = item[1]["x"],item[1]["y"],item[1]["z"]
        # qKey = mercator.QuadTree(tx, ty, tz)
        url = googleMapURLs[url_index].format(**{'x':tx2,"y":ty2,"z":tz2})
        save_path = os.path.join(download_path,str(tz),str(tx),"{}.png".format(str(ty)))
        # print(i,len(data),i/len(data))
        list_url.append(url)
        list_save_path.append(save_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = list(tqdm(executor.map(download_tile_esri, list_url,list_save_path), total=len(list_url)))
        download_tile_esri(url,save_path)
    return True