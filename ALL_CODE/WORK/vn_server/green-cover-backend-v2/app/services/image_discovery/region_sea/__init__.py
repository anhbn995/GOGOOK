from app.services.image_discovery.region_sea.sentinel1_service import DiscoverySentinel1

class ImageDiscoverySeaFactory():
    def __init__(self, code):
        self.code = code

    def create_discovery_image(self):
        if self.code == 'sentinel1':
            return DiscoverySentinel1()
    