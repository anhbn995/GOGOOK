import itertools

from app.services.image_discovery.region_eu.image_discovery_eu_service import ImageDiscoveryEu 
from datetime import datetime

class DiscoveryLandsat8(ImageDiscoveryEu):
    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs
    
    def prepare(self):
        self.from_date = datetime.strptime(self._kwargs.get('from_date'), '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y%m%d')
        self.to_date = datetime.strptime(self._kwargs.get('to_date'), '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y%m%d')
        self.season = self._kwargs.get('season')
        self.geometry = self._kwargs.get('geometry')
        self.orbitdirection = self._kwargs.get('orbitdirection')
        self.cloud_cover = self._kwargs.get('cloud_cover') * 100 if self._kwargs.get('cloud_cover') <= 1 else self._kwargs.get('cloud_cover')  
        self.product = 'Landsat8'

    def transform(self):
        result = []
        for list_item in list(itertools.chain(*self.list_items)):
            for feature in list_item:
                result.append({
                    'id': feature['id'],
                    'geometry': feature['geometry'],
                    'properties': {
                        'subdomains': [0, 1, 2, 3],
                        'tile_url': None,
                        'acquired': feature['properties']['published'],
                        'cloud_cover': feature['properties']['cloudCover'] / 100,
                        'product_id': feature['properties']['title']
                    },
                    'thumbnail': feature['properties']['thumbnail'],
                })
        return result