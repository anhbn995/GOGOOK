from abc import abstractmethod

from app.services.image_discovery.image_discovery_service import ImageDiscoveryService

class ImageDiscoverySea(ImageDiscoveryService):
    from_date = None
    to_date = None
    season = None
    geometry = None
    orbitdirection = None
    list_items = []

    def __init__(self):
        super().__init__()

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def discovery_image(self):
        pass

    @abstractmethod
    def transform(self):
        pass
