import geopandas as gpd
import pandas as pd
import time
import numpy as np
import os
import json
from shapely.geometry import Polygon
from utils.tileutils import ( getExtentOfPolygon, getGridMatrix,
                            getGridBound, intersect, tileToGoogleTile, 
                            bound_to_polygon,download_bing_tile,download_esri_tile)
from mbutil import disk_to_mbtiles
import shutil
from retrying import retry
import requests
from requests_toolbelt.multipart import encoder
from pebble import concurrent
# function to get unique values
@concurrent.process(timeout=3600)
def upload_and_remove_process(upload_mbtiles,remove_tiles_dir):
    session = requests.Session()
    with open(upload_mbtiles, 'rb') as f:
        form = encoder.MultipartEncoder({
           'files[]': (os.path.basename(upload_mbtiles), f, "application/octet-stream")})
        headers = {"Prefer": "respond-async", "Content-Type": form.content_type}
        resp = session.post("https://apiv2.eofactory.ai:3443/api/bing_indo", headers=headers, data=form)
    session.close()
    # files = [('files[]', open(upload_mbtiles, 'rb'))]
    # r = requests.post('https://apiv2.eofactory.ai:3443/api/bing_indo', files=files)
    json_data = json.loads(resp.text)
    if json_data["message"] == 'Successful':
        os.remove(upload_mbtiles)
        shutil.rmtree(remove_tiles_dir)
    return True

@retry(
    wait_fixed=10000
)
def upload_and_remove(upload_mbtiles,remove_tiles_dir):
    result = upload_and_remove_process(upload_mbtiles,remove_tiles_dir).result()
    return True

def create_list_mbtiles(dowwnload_maps_dir):
    list_id = []
    for file in os.listdir(dowwnload_maps_dir):
        if file.endswith(".mbtiles"):
            list_id.append(file[:-8])
    # list_id.sort()
    return list_id
@retry(
    wait_fixed=1000
)
def check_exists(name_tiles):
    r = requests.get("https://apiv2.eofactory.ai:3443/api/bing_indo/file_exists?name={}.mbtiles".format(name_tiles))
    data = json.loads(r.text)
    return data["data"]["is_exists"]
def split_shape():
    input_shape = "D:\Kinabatangan_2.geojson"
    output_shape = "E:\download_tiles2\Kinabatangan_grid.geojson"
    boundary_shp = gpd.read_file(input_shape)
    bound = boundary_shp.iloc[0].geometry
    extend = bound.bounds
    zoomLevel = 11
    offset = getGridMatrix(extend, zoomLevel=zoomLevel)
    gridMatrix = np.zeros((offset[3] - offset[1] + 1, offset[2] - offset[0] + 1, 3),dtype=np.uint32)
    height,width,_ = gridMatrix.shape
    print(gridMatrix.shape)
    for i in range(height):
        for j in range(width):
            gridMatrix[i,j] = (offset[0] + j, offset[3] - i, zoomLevel)
            
    boundGridMatrix = np.reshape(np.apply_along_axis(getGridBound, 1, gridMatrix.reshape(width * height, 3)), (height, width, 4))
    
    intersectGridMatrix = np.reshape(np.apply_along_axis(intersect, 1, boundGridMatrix.reshape(width * height, 4), bound), (height, width))
    
    googleGridMatrix = np.reshape(np.apply_along_axis(tileToGoogleTile, 1, gridMatrix.reshape(width * height, 3)), (height, width, 3)).astype(np.uint32)
    
    data = []
    for i in range(height):
        for j in range(width):
            if intersectGridMatrix[i][j]:
                x1,y1,z1=gridMatrix[i][j]
                x2,y2,z2=googleGridMatrix[i][j]
                data.append([bound_to_polygon(boundGridMatrix[i][j]),{"x":int(x1),"y":int(y1),"z":int(z1)},{"x":int(x2),"y":int(y2),"z":int(z2)}])
    df_polygon = pd.DataFrame(data, columns=['geometry', 'tile_tms',"tile_google"])
    gdf_polygon = gpd.GeoDataFrame(df_polygon, geometry='geometry', crs=boundary_shp.crs)
    gdf_polygon.to_file(output_shape,driver='GeoJSON')  
    return True      
    
if __name__ == '__main__':

    # split_shape()
    with open("./config.json") as json_file:
        config = json.loads(json_file.read())
    a = time.time()
    shape_path = config["shape_path"]
    out_path_tiles = config["output_path"]
    zoomLevel = config["zoomLevel"]
    out_name = config["name"]
    id_computer = config["id_computer"]
    total_computer = config["total_computer"]
    map_name = config["map"]
    save_point = config["save_point"]
    resume = save_point
    if not os.path.exists(out_path_tiles):
        os.makedirs(out_path_tiles)
    boundary_shp = gpd.read_file(shape_path)
    if total_computer <=1:
        step = len(boundary_shp)
        boundary_shp_split = boundary_shp
    else:
        step = len(boundary_shp)//(total_computer)
        split_list = list(range(0,len(boundary_shp),step))
        if split_list[-1] != len(boundary_shp):
            split_list[-1] = len(boundary_shp)
        index1,index2 = split_list[id_computer-1],split_list[id_computer]
        boundary_shp_split = boundary_shp.iloc[index1:index2]
    for k in range(save_point,len(boundary_shp_split)):
        resume = resume+1
        print("start download bound {}/{}".format(k+1,len(boundary_shp_split)))
        bound = boundary_shp_split.iloc[k].geometry
        extend = bound.bounds
        if 'tile_tms' in boundary_shp_split.columns:
            x_tiles = boundary_shp_split.iloc[k].tile_tms["x"]
            y_tiles = boundary_shp_split.iloc[k].tile_tms["y"]
            name_tiles = out_name+ "_{}_{}".format(x_tiles,y_tiles)
        else:
            name_tiles = out_name+ "_{}".format(str(k))
        # check_ex = check_exists(name_tiles)
        check_ex = False
        if check_ex == True:
            pass
        else:
            offset = getGridMatrix(extend, zoomLevel=zoomLevel)
            gridMatrix = np.zeros((offset[3] - offset[1] + 1, offset[2] - offset[0] + 1, 3),dtype=np.uint32)
            height,width,_ = gridMatrix.shape
            print(gridMatrix.shape)
            for i in range(height):
                for j in range(width):
                    gridMatrix[i,j] = (offset[0] + j, offset[3] - i, zoomLevel)
                    
            boundGridMatrix = np.reshape(np.apply_along_axis(getGridBound, 1, gridMatrix.reshape(width * height, 3)), (height, width, 4))
            
            intersectGridMatrix = np.reshape(np.apply_along_axis(intersect, 1, boundGridMatrix.reshape(width * height, 4), bound), (height, width))
            
            googleGridMatrix = np.reshape(np.apply_along_axis(tileToGoogleTile, 1, gridMatrix.reshape(width * height, 3)), (height, width, 3)).astype(np.uint32)
            
            data = []
            for i in range(height):
                for j in range(width):
                    if intersectGridMatrix[i][j]:
                        x1,y1,z1=gridMatrix[i][j]
                        x2,y2,z2=googleGridMatrix[i][j]
                        data.append([bound_to_polygon(boundGridMatrix[i][j]),{"x":int(x1),"y":int(y1),"z":int(z1)},{"x":int(x2),"y":int(y2),"z":int(z2)}])
            dowwnload_maps_dir = os.path.join(out_path_tiles,map_name)
            if not os.path.exists(dowwnload_maps_dir):
                os.makedirs(dowwnload_maps_dir)
            if map_name =="esri":
                download_esri_tile(data,os.path.join(dowwnload_maps_dir,name_tiles))
                kwargs = {"schema":"tms","image_format":"png","grid_callback":'grid',"silent":False,"do_compression":False}
                output_mbtiles = os.path.join(dowwnload_maps_dir,"{}.mbtiles".format(name_tiles))
                if os.path.isfile(output_mbtiles):
                    os.remove(output_mbtiles)
                tiles_dir = os.path.join(dowwnload_maps_dir,name_tiles)
                disk_to_mbtiles(tiles_dir,output_mbtiles,**kwargs)
            else:
                download_bing_tile(data,os.path.join(dowwnload_maps_dir,name_tiles))
                print(time.time() -a)
                kwargs = {"schema":"tms","image_format":"png","grid_callback":'grid',"silent":False,"do_compression":False}
                output_mbtiles = os.path.join(dowwnload_maps_dir,"{}.mbtiles".format(name_tiles))
                if os.path.isfile(output_mbtiles):
                    os.remove(output_mbtiles)
                tiles_dir = os.path.join(dowwnload_maps_dir,name_tiles)
                disk_to_mbtiles(tiles_dir,output_mbtiles,**kwargs)

            # list_mbtiles = create_list_mbtiles(dowwnload_maps_dir)
            # if k < (len(boundary_shp_split)-1) and len(list_mbtiles) >= 5:
            #     print("upload start")
            #     while len(list_mbtiles)>=3:
            #         upload_mbtiles = list_mbtiles[0]
            #         upload_mbtilespath = os.path.join(dowwnload_maps_dir,upload_mbtiles+".mbtiles")
            #         remove_tiles_dir = os.path.join(dowwnload_maps_dir,upload_mbtiles)
            #         shutil.copy(upload_mbtilespath, r"W:\malaysia_Kinabatangan")
            #         os.remove(upload_mbtilespath)
            #         shutil.rmtree(remove_tiles_dir)
            #         # upload_and_remove(upload_mbtilespath,remove_tiles_dir)
            #         print("upload {}".format(upload_mbtilespath))
            #         list_mbtiles.pop(0)
            # elif k == (len(boundary_shp_split)-1):
            #     print("upload start")
            #     while len(list_mbtiles)>0:
            #         upload_mbtiles = list_mbtiles[0]
            #         upload_mbtilespath = os.path.join(dowwnload_maps_dir,upload_mbtiles+".mbtiles")
            #         remove_tiles_dir = os.path.join(dowwnload_maps_dir,upload_mbtiles)
            #         shutil.copy(upload_mbtilespath, r"W:\malaysia_Kinabatangan")
            #         os.remove(upload_mbtilespath)
            #         shutil.rmtree(remove_tiles_dir)
            #         # upload_and_remove(upload_mbtilespath,remove_tiles_dir)
            #         print("upload {}".format(upload_mbtilespath))
            #         list_mbtiles.pop(0)
            
# files = [('files[]', open(r"C:\Users\lamng\Desktop\tile_z11_1565_1057.mbtiles", 'rb')), ('files[]', open(r"C:\Users\lamng\Desktop\tile_z11_1565_1057_2.mbtiles", 'rb'))]
# r = requests.post('https://apiv2.eofactory.ai:3443/api/bing_indo', files=files)
        
    # df_polygon = pd.DataFrame(data, columns=['geometry', 'tile_tms',"tile_google"])            
    # gdf_polygon = gpd.GeoDataFrame(df_polygon, geometry='geometry', crs=boundary_shp.crs)
    # gdf_polygon.to_file(r"/media/skymap/Learnning/public/test_simplify/tiles2_indo.geojson",driver='GeoJSON')