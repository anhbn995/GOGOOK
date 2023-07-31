import shapely.wkt
import json
import threading
import os

from datetime import datetime
from geodaisy import GeoObject
from shapely.geometry import Polygon, MultiPolygon, box

from app.services.image_discovery.region_sea.image_discovery_sea_service import ImageDiscoverySea 
from app.utils.geometry import split_geom_to_grids
from app.services.image_discovery.region_sea.image_discovery_sea_thread import ImageDiscoverySeaThread

class DiscoverySentinel1(ImageDiscoverySea):
    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs
    
    def prepare(self):
        self.from_date = self._kwargs.get('from_date')
        self.to_date = self._kwargs.get('to_date')
        self.season = self._kwargs.get('season')
        self.geometry = self._kwargs.get('geometry')
        self.orbitdirection = self._kwargs.get('orbitdirection')

    def discovery_image(self):
        list_items = []
        if not self.season:
            products = self._sentinelAPIQuery(self.from_date, self.to_date)
            list_items.append(products)
        else:
            start_season = self.season['start']
            end_season = self.season['end']
            years = self.season['years']
            for year in years:
                start_date = f'{year}{start_season}'
                end_date = f'{year}{end_season}'
                products = self._sentinelAPIQuery(start_date, end_date, orbitdirection)
                list_items.append(products)    
        self.list_items = list_items   
        
    def _sentinelAPIQuery(self, start_date, end_date):
        all_product = {}
        for feature in self.geometry['features']:
            try:
                sub_features = iter(split_geom_to_grids(shapely.geometry.shape(feature['geometry']), 2))
                list_lock = threading.Lock()
                out_lock = threading.Lock()
                threads = []

                for i in range(int(os.cpu_count()/3)):
                    thread = ImageDiscoverySeaThread(
                        self.orbitdirection, 
                        start_date, 
                        end_date, 
                        list_sub_futures=sub_features,
                        list_lock=list_lock, 
                        out_lock=out_lock, 
                        output=all_product)
                    threads.append(thread)
                    thread.start()
                for thread in threads:
                    thread.join()
            except Exception as e:
                print(e)
                continue
        return all_product

    def _get_timefrom_title(self, title):
        list_str = title.split('_')
        sub_str = list_str[4]
        year = (sub_str[:4])
        month = (sub_str[4:6])
        day = (sub_str[6:8])
        return  year,month,day

    def transform(self):
        result = []
        for list_item in self.list_items:
            for k, p in list_item.items():
                title = p['title']
                geom_wkt = p['footprint']
                geom_type = shapely.wkt.loads(geom_wkt).geom_type
                if geom_type == 'MultiPolygon':
                    geo_obj = GeoObject(MultiPolygon(shapely.wkt.loads(geom_wkt)))
                else:
                    geo_obj = GeoObject(Polygon(shapely.wkt.loads(geom_wkt)))
                geom_geojson = json.loads(geo_obj.geojson())
                year,month,day = self._get_timefrom_title(title)
                _item = {
                    'title': title,
                    'geom': geom_geojson,
                    'date': '{}-{}-{}'.format(year,month,day)
                }
                if '_1SDV' in title:
                    result.append(_item)
        return result