import itertools

from app.services.image_discovery.region_eu.image_discovery_eu_service import ImageDiscoveryEu 
from datetime import datetime

class DiscoverySentinel1(ImageDiscoveryEu):
    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs
    
    def prepare(self):
        self.from_date = self._kwargs.get('from_date')
        self.to_date = self._kwargs.get('to_date')
        self.season = self._kwargs.get('season')
        self.geometry = self._kwargs.get('geometry')
        self.orbitdirection = self._kwargs.get('orbitdirection')
        self.product = 'Sentinel1'

    def transform(self):
        result = []
        for list_item in list(itertools.chain(*self.list_items)):
            for feature in list_item:
                try:
                    date = datetime.strptime(feature['properties']['startDate'], '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y-%m-%d')
                except:
                    date = datetime.strptime(feature['properties']['startDate'], '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')
                if '_1SDV' in feature['properties']['title']:
                    result.append({
                        'date': date,
                        'geom': feature['geometry'],
                        'title': feature['properties']['title'].split('.')[0],
                        'thumbnail': feature['properties']['thumbnail']
                    })
        return result