import pdb
import uuid

import requests
from config.default import HOSTED_ENDPOINT
from config.default import ROOT_DATA_FOLDER
from pathlib import Path
import geopandas as gpd
import os
import ast
from shapely.geometry import Polygon, MultiPolygon

def remove_invalid_gdf(gdf):
    gdf['geometry'] = gdf.buffer(0)

    for ind, g in enumerate(gdf['geometry']):
        if not type(g) in [Polygon, MultiPolygon] or not g.is_valid or g.is_empty:
            if not type(g) in [Polygon, MultiPolygon] or not g.is_valid or g.is_empty:
                gdf = gdf.drop(ind)
    return gdf


def store_vectors(payload):
    url = f'{HOSTED_ENDPOINT}/internal/vectors'

    print(url)
    r = requests.post(url, json=payload)
    if not r.ok:
        # print(r.json())
        raise Exception('Fail to upload result')


def store_cloud(file_path,month, year, src, aoi):

    fid_vector = uuid.uuid4()
    dir_path = f'{ROOT_DATA_FOLDER}/data/{year}/{month}/vectors'
    payload = []

    gdf = gpd.GeoDataFrame.from_features(aoi["geometry"])
    gdf = remove_invalid_gdf(gdf)
    if not gdf.crs:
        gdf.crs = {"init": "epsg:4326"}
    gdf = gdf.to_crs('EPSG:4326')

    df = gpd.read_file(file_path)
    df = df.to_crs('EPSG:4326')
    clip = gpd.clip(df, gdf)

    print(os.stat(file_path).st_size)
    abs_path = f'{dir_path}/{fid_vector.hex}.geojson'
    if not clip.empty:
        clip.to_file(abs_path, driver="GeoJSON")
        payload.append({
            'id': fid_vector.hex,
            'name': f'{year}_{month}_cloud_cover',
            'path': f"/{os.path.relpath(abs_path, ROOT_DATA_FOLDER)}",
            'month': month,
            'year': year,
            'type': 'cloud',
            'src': src['key'],
            'geometry': ast.literal_eval(clip.to_json()),
            'aoi_id': aoi['id']
        })
        store_vectors(payload)
    else:
        print(f'month: {month},year:{year},aoi: {aoi["id"]} not intersection')



if __name__ == '__main__':

    store_cloud('/home/geoai/geoai_data_test/data_npark/Sentinel/cloud/2021_T1.shp',2021,'01','Sentiel',14)
    year = 2021
    # list_geojson = [
    #     {
    #         'path': '/home/thang/Desktop/shp/T1.shp',
    #         'month': '01',
    #     }
    # ]

    list_geojson = []
    folder_vector = '/home/thang/Desktop/projects/npark/vectors/shp_4326'
    list_kml = []
    payload = []
    for i in range(1, 9):
        for path in Path(folder_vector).rglob(f'*{i}.shp'):
            file_path = str(path.resolve())
            fid_vector = uuid.uuid4()

            if len(str(i)) == 1:
                month = f'0{str(i)}'
            else:
                month = str(i)
            dir_path = f'{ROOT_DATA_FOLDER}/data/{year}/{month}/vectors'

            try:
                df = gpd.read_file(file_path)
                print(os.stat(file_path).st_size)
                abs_path = f'{dir_path}/{fid_vector.hex}.geojson'
                df.to_file(abs_path, driver="GeoJSON")
                payload.append({
                    'id': fid_vector.hex,
                    'name': f'{year}_{month}_cloud',
                    'path': f"/{os.path.relpath(abs_path, ROOT_DATA_FOLDER)}",
                    'month': month,
                    'year': year,
                    'type': 'cloud',
                    'src': 'Sentinel',
                    'geometry': ast.literal_eval(df.to_json()),
                    'aoi_id': 14,

                })

            except Exception as e:
                print(e)

    # for item in list_geojson:
    #     with open(item['path']) as f:
    #         gj = geojson.load(f)
    #     payload.append({
    #         'name': item['name'],
    #         'geometry': gj
    #     })
    store_vectors(payload)
