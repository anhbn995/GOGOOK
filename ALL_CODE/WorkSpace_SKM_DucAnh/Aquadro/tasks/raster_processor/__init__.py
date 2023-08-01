
from abc import abstractmethod

from lib.tools.raster.services.io import write_symbology
from lib.tools.raster.services.symbology import get_symbology
from lib.tools.raster.services.tile import get_thumbnail, get_tile
from tasks import TaskExecutor
from osgeo import gdal
from lib.tools.raster.services.stretch import stretch
from lib.tools.raster.services.statistic import calculate_metadata


class RasterProcessor(TaskExecutor):
    metadata = {}
    tile_url = ''
    thumbnail = ''
    feature_id = None

    def __init__(self, task_id: int, token: str, group: str, target_id: int, payload: dict):
        super().__init__(task_id, token, group, target_id, payload)

    @property
    def output_path(self):
        return f'{self.output_dir}/{self.file_id}.tif'

    @property
    def output_pscheme_path(self):
        return f'{self.output_dir}/{self.file_id}.json'

    @abstractmethod
    def run_task(self):
        pass

    def on_success(self):
        self.store_result({
            "tile_url": self.tile_url,
            'path': self.relative_output_path,
            'metadata': self.metadata,
            'thumbnail': self.thumbnail,
            'crs': self.metadata['crs'],
            "size": self.get_file_size(self.output_path),
            'bbox': self.metadata.get('bbox'),
            'file_id': self.file_id,
            'feature_id': self.feature_id,

        })

    def calculate_percentile_if_neccesary(self):
        if not self.is_byte_output():
            stretch(self.output_path, self.output_pscheme_path)

    def is_byte_output(self):
        raster = gdal.Open(self.output_path)
        band = raster.GetRasterBand(1)
        data_type = band.DataType
        return data_type == 1

    def complete_task(self):
        self.calculate_percentile_if_neccesary()
        self.calculate_metadata()
        self.save_symbology()
        self.generate_tile_url()
        # self.generate_thumbnail()

    def save_symbology(self):
        self.metadata['symbology'] = get_symbology(
            self.output_path, self.metadata)
        write_symbology(self.output_path, self.metadata['symbology'])
        self.metadata['symbology']['nodata'] = self.metadata['nodata']

    def calculate_metadata(self):
        self.metadata = calculate_metadata(self.output_path)

    def generate_tile_url(self):
        self.tile_url = get_tile(
            self.metadata['symbology'], self.group, self.target_id, self.file_id)

    def generate_thumbnail(self):
        self.thumbnail = get_thumbnail(
            self.metadata['symbology'], self.group, self.target_id, self.file_id)
