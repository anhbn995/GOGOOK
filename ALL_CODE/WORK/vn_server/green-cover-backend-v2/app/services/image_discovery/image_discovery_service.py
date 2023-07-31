from abc import ABC, abstractmethod
from app.utils.response import success

class ImageDiscoveryService(ABC):
    def __init__(self):
        super().__init__()

    @classmethod
    def execute(cls, **kwargs):
        instance = cls(**kwargs)
        try:
            instance.prepare()
            instance.discovery_image()
            return success(instance.transform())
        except Exception as exception:
            print(exception)

    @abstractmethod
    def prepare(self):
        pass   

    @abstractmethod
    def discovery_image(self):
        pass   

    @abstractmethod
    def transform(self):
        pass            
