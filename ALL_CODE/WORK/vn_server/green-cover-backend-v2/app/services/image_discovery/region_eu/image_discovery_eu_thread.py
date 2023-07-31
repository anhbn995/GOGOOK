import requests
import urllib

from threading import Thread
from app.services.image_discovery.image_discovery_thread import ImageDiscoveryThread

class ImageDiscoveryEuThread(ImageDiscoveryThread):
    def __init__(self, query, product, **kwargs):
        super().__init__(**kwargs)
        self._query = query
        self._product = product

    def send_discovery_api(self, item):
        url = f"https://finder.creodias.eu/resto/api/collections/{self._product}/search.json?geometry={item}&" + urllib.parse.urlencode(self._query, doseq=False, safe='[,]', quote_via=urllib.parse.quote)
        response = requests.get(url.replace('%20', ''))
        if response.status_code != 200:
            return None
        return response.json()['features']