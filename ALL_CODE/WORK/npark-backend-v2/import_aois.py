import requests
from config.default import HOSTED_ENDPOINT
import geojson
from pathlib import Path
import geopandas as gpd
import os
import ast


def store_aois(payload):
    url = f'{HOSTED_ENDPOINT}/internal/aois'

    print(url)
    r = requests.post(url, json=payload)
    if not r.ok:
        raise Exception('Fail to upload result')


if __name__ == '__main__':
    # list_geojson = [
    #     {
    #         'path': '/home/thang/Desktop/2_aoi_singapore.geojson',
    #         'name': 'AOI SELECTED'
    #
    #     },
    #     {
    #         'path': '/home/thang/Downloads/new_aoi.geojson',
    #         'name': 'AOI Entire Singapore',
    #
    #     }
    # ]
    list_geojson = []
    folder_vector = '/home/thang/Desktop/projects/npark/vectors'
    list_kml = []
    payload = []
    gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
    # for path in Path(folder_vector).rglob('*'):
    kml_path = '/home/geoai/geoai_data_test2/8_GreenCover_GeoMineraba/1_Data_origin/SAMPLE DATA GEOMINERBA/Boundry Locus Pemantauan Citra Sept-Des 2021_Malut/PT_ZHONG,_PT_BPN_dan_PT_DRI_.geojson'
    try:
        df = gpd.read_file(kml_path)
        df = df.explode()
        print(os.stat(kml_path).st_size)
        payload.append({
            'name': 'PT_ZHONG,_PT_BPN_dan_PT_DRI',
            'geometry': ast.literal_eval(df.to_json().replace("null", '""'))
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

    store_aois(payload)
