

from abc import abstractmethod


class Tile:
    @abstractmethod
    def render(self, z: int, x: int, y: int):
        pass

    @abstractmethod
    def get_json(self, group, target_id, file_id):
        pass


class Coordinates():
    def __init__(self, z: int, x: int, y: int):
        self.z = int(z)
        self.x = int(x)
        self.y = int(y)
