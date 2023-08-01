
from config.storage import ROOT_DATA_FOLDER
from osgeo import gdal
from lib.tools.raster.services.translate import translate
from tasks.raster_processor import RasterProcessor
from math import ceil


class UpScaleProcessor(RasterProcessor):
    def __init__(self, task_id: int, token: str, group: str, target_id: int, payload: dict):
        super().__init__(task_id, token, group, target_id, payload)
        self.input_path = ROOT_DATA_FOLDER + payload.get('input_path')
        self.params = payload.get('params') or {}
        self.level = int(self.params.get('level')
                         ) if self.params.get('level') else 2

    def run_task(self):
        tmp_path = f'{self.temp_dir}/image.tif'
        ds = gdal.Open(self.input_path)
        xsize, ysize = ds.RasterXSize, ds.RasterYSize
        result_xsize, result_ysize = ceil(
            float(xsize)/self.level), ceil(float(ysize)/self.level)
        cmd = f'-r average -ts {result_xsize} {result_ysize} -wm 2048 -multi -wo NUM_THREADS=ALL_CPUS ' \
              '-co BLOCKXSIZE=256 -co BLOCKYSIZE=256 -co TILED=YES -co NUM_THREADS=ALL_CPUS -co COMPRESS=LZW'
        options = gdal.WarpOptions(gdal.ParseCommandLine(cmd))
        gdal.Warp(tmp_path, self.input_path, options=options)
        translate(tmp_path, self.output_path)