from app.services.image_discovery.region_eu.sentinel1_service import DiscoverySentinel1
from app.services.image_discovery.region_eu.sentinel2_service import DiscoverySentinel2
from app.services.image_discovery.region_eu.landsat8_service import DiscoveryLandsat8

class ImageDiscoveryEuFactory():
    def __init__(self, code):
        self.code = code

    def create_discovery_image(self):
        if self.code == 'sentinel1_eu':
            return DiscoverySentinel1()
        if self.code == 'sentinel2_eu':
            return DiscoverySentinel2()
        if self.code == 'landsat8_eu':
            return DiscoveryLandsat8()