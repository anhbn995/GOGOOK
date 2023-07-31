import shapely.wkt
import threading
import os

from datetime import datetime
from abc import abstractmethod

from app.services.image_discovery.image_discovery_service import ImageDiscoveryService
from app.utils.geometry import split_geom_to_grids
from app.services.image_discovery.region_eu.image_discovery_eu_thread import ImageDiscoveryEuThread
class ImageDiscoveryEu(ImageDiscoveryService):
    from_date = None
    to_date = None
    season = None
    geometry = None
    orbitdirection = None
    cloud_cover = ''
    processing_level = ''
    product = None
    list_items = []

    def __init__(self):
        super().__init__()

    @abstractmethod
    def prepare(self):
        pass

    def discovery_image(self):
        list_items = []
        if not self.season:
            list_items.append(self._query_image(self.from_date, self.to_date))
        else:
            start_season = self.season['start']
            end_season = self.season['end']
            years = self.season['years']
            for year in years:
                start_date = f'{year}{start_season}'
                end_date = f'{year}{end_season}'  
                list_items.append(self._query_image(start_date, end_date))
        self.list_items = list_items

    def _query_image(self, start_date, end_date):
        try:
            query = {
                'status': 'all',
                'dataset': 'ESA-DATASET',
                'maxRecords': 1000,
                'sortParam': 'startDate',
                'sortOrder': 'descending',
                'startDate': datetime.strptime(start_date, '%Y%m%d').strftime('%Y-%m-%d'),
                'completionDate': datetime.strptime(end_date, '%Y%m%d').strftime('%Y-%m-%d'),
                'orbitDirection': self.orbitdirection.lower() if self.orbitdirection else '',
                'processingLevel': self.processing_level,
                'cloudCover': [0,int(self.cloud_cover)] if self.cloud_cover else ''
            }
            all_product = []
            for feature in self.geometry['features']:
                try:
                    sub_features = iter(split_geom_to_grids(shapely.geometry.shape(feature['geometry']), 2))
                    list_lock = threading.Lock()
                    out_lock = threading.Lock()
                    threads = []

                    for i in range(int(os.cpu_count()/3)):
                        thread = ImageDiscoveryEuThread(
                            query, 
                            self.product,
                            list_sub_futures=sub_features,
                            list_lock=list_lock, 
                            out_lock=out_lock, 
                            output=all_product
                        )
                        threads.append(thread)
                        thread.start()
                    for thread in threads:
                        thread.join()
                except Exception as e:
                    continue
            return all_product
        except Exception as e:
            print(e)

    @abstractmethod
    def transform(self):
        pass
