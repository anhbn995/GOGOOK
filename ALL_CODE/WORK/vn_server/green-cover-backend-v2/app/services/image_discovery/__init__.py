from app.services.image_discovery.region_eu import ImageDiscoveryEuFactory
from app.services.image_discovery.region_sea import ImageDiscoverySeaFactory

class ImageDiscoveryFactory():
    def search(self, **kwargs):
        code = kwargs.get('code')
        if '_eu' in code:
            image_discovery = ImageDiscoveryEuFactory(code)
        else:
            image_discovery = ImageDiscoverySeaFactory(code)
        return image_discovery.create_discovery_image().execute(**kwargs)