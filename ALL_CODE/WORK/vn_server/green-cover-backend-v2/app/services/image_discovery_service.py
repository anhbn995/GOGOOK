import ogr
import json
import shapely.wkt
import requests
import urllib
from datetime import datetime

from geodaisy import GeoObject
from sentinelsat import SentinelAPI, geojson_to_wkt, read_geojson
from shapely.geometry import Polygon, MultiPolygon, box

from app.utils.response import success
from config.default import SENTINEL_API_USER, SENTINEL_API_PASSWORD, SENTINEL_API_URL
from app.utils.geometry import split_geom_to_grids

class ImageDiscoveryService():

    def sen1_search(self, geometry, image_types, orbitdirection=None, season=None, date_range=None, **kwargs):
        def ordered_dict_2_list(products):
            list_item = []
            from shapely import wkt
            for k, p in products.items():
                title = p['title']
                geom_wkt = p['footprint']
                geom_type = wkt.loads(geom_wkt).geom_type
                if geom_type == 'MultiPolygon':
                    geo_obj = GeoObject(MultiPolygon(shapely.wkt.loads(geom_wkt)))
                else:
                    geo_obj = GeoObject(Polygon(shapely.wkt.loads(geom_wkt)))
                geom_geojson = json.loads(geo_obj.geojson())
                year,month,day = self.get_timefrom_title(title)
                _item = {
                    'title': title,
                    'geom': geom_geojson,
                    'date': '{}-{}-{}'.format(year,month,day)
                }
                list_item.append(_item)
            return list_item
        
        def sentinelAPIQuery(start_date, end_date, orbitdirection):
            all_product = {}
            for feature in geometry['features']:
                try:
                    sub_features = split_geom_to_grids(shapely.geometry.shape(feature['geometry']), 2)
                    print(len(sub_features))
                    for key, sub_feature in enumerate(sub_features):
                        try:
                            if orbitdirection and orbitdirection.upper() in ['ASCENDING', 'DESCENDING']: 
                                product = api.query(
                                    sub_feature,
                                    date=(start_date, end_date),
                                    platformname='Sentinel-1',
                                    producttype='GRD',
                                    orbitdirection=orbitdirection.upper(),
                                    filename='S1A_IW*'
                                )
                            else:
                                product = api.query(
                                    sub_feature,
                                    date=(start_date, end_date),
                                    platformname='Sentinel-1',
                                    producttype='GRD',
                                    filename='S1A_IW*',
                                )
                            all_product.update(product)
                        except Exception as e:
                            print(key)
                            print(e)
                            continue
                except Exception as e:
                    print(e)
                    continue
            return all_product
        
        api = SentinelAPI(SENTINEL_API_USER, SENTINEL_API_PASSWORD, SENTINEL_API_URL)
        list_item = []
        if date_range:
            start_date = date_range[0]
            end_date = date_range[1]
            # Search Sentinel
            products = sentinelAPIQuery(start_date, end_date, orbitdirection)
            list_item += ordered_dict_2_list(products)
        else:
            start_season = season.get('start')
            end_season = season.get('end')
            years = season.get('years')
            for year in years:
                start_date = f'{year}{start_season}'
                end_date = f'{year}{end_season}'
                query_result = sentinelAPIQuery(start_date, end_date, orbitdirection)
                list_item += ordered_dict_2_list(query_result)
        result = []
        for item in list_item:
            if '_1SDV' in item.get('title'):
                result.append(item)
        return success(result)
    
    def sen1_eu_search(self, geometry, image_types, orbitdirection=None, season=None, date_range=None, **kwargs):
        list_items = []
        if date_range:
            start_date = date_range[0]
            end_date = date_range[1]
            
            list_items.append(self.query_sentinel1_eu(start_date, end_date, geometry, orbitdirection))
        else:
            start_season = season.get('start')
            end_season = season.get('end')
            years = season.get('years')
            for year in years:
                start_date = f'{year}{start_season}'
                end_date = f'{year}{end_season}'  
                list_items.append(self.query_sentinel1_eu(start_date, end_date, geometry, orbitdirection))
        result = []

        import itertools
        for list_item in list(itertools.chain(*list_items)):
            for feature in list_item:
                try:
                    date = datetime.strptime(feature['properties']['startDate'], '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y-%m-%d')
                except:
                    date = datetime.strptime(feature['properties']['startDate'], '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')
                result.append({
                    'date': date,
                    'geom': feature['geometry'],
                    'title': feature['properties']['title'].split('.')[0],
                    'thumbnail': feature['properties']['thumbnail']
                })
        return success(result)

    def get_timefrom_title(self, title):
        list_str = title.split('_')
        sub_str = list_str[4]
        year = (sub_str[:4])
        month = (sub_str[4:6])
        day = (sub_str[6:8])
        return  year,month,day

    # def get_aoi_bounds(self, aoi):
    #     geoms = []
    #     for feature in aoi['features']:
    #         geom = shapely.geometry.shape(feature['geometry'])
    #         geom = shapely.wkt.loads(shapely.wkt.dumps(geom, rounding_precision=5))
    #         geoms.append(geom)
    #     return shapely.geometry.GeometryCollection(geoms).wkt

    # def simplify_feature(self, feature):
    #     geom = shapely.geometry.shape(feature['geometry'])
    #     geom = shapely.wkt.dumps(geom.simplify(0.05), rounding_precision=4)
    #     return geom

    def query_sentinel1_eu(self, start_date, end_date, geometry, orbitdirection):
        start_date = datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d') 
        all_product = [] 
        for feature in geometry['features']:
            sub_features = split_geom_to_grids(shapely.geometry.shape(feature['geometry']), 2)
            for sub_feature in sub_features:
                orbitdirection = orbitdirection.lower() if orbitdirection else ''
                response = requests.post(f"http://finder.creodias.eu/resto/api/collections/Sentinel1/search.json?maxRecords=1000&sortParam=startDate&sortOrder=descending&startDate={start_date}&completionDate={end_date}&geometry={sub_features}&orbitDirection={orbitdirection}")
                if response.status_code == 200:
                    all_product.append(response.json()['features'])
        return all_product

