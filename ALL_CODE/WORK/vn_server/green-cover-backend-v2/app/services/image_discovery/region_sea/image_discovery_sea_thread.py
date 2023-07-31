import requests
import urllib

from threading import Thread
from app.services.image_discovery.image_discovery_thread import ImageDiscoveryThread
from config.default import SENTINEL_API_USER, SENTINEL_API_PASSWORD, SENTINEL_API_URL
from sentinelsat import SentinelAPI, geojson_to_wkt, read_geojson

class ImageDiscoverySeaThread(ImageDiscoveryThread):
    def __init__(self, orbitdirection, start_date, end_date, **kwargs):
        super().__init__(**kwargs)
        self._orbitdirection = orbitdirection
        self._start_date = start_date
        self._end_date = end_date

    def send_discovery_api(self, item):
        api = SentinelAPI(SENTINEL_API_USER, SENTINEL_API_PASSWORD, SENTINEL_API_URL)
        try:
            if self._orbitdirection and self._orbitdirection.upper() in ['ASCENDING', 'DESCENDING']: 
                product = api.query(
                    item,
                    date=(self._start_date, self._end_date),
                    platformname='Sentinel-1',
                    producttype='GRD',
                    orbitdirection=self._orbitdirection.upper(),
                    filename='S1A_IW*'
                )
            else:
                product = api.query(
                    item,
                    date=(self._start_date, self._end_date),
                    platformname='Sentinel-1',
                    producttype='GRD',
                    filename='S1A_IW*',
                )
            return product
        except Exception as e:
            return None
            print(e)