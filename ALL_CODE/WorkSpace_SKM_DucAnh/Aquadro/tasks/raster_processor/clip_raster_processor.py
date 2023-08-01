
import json
from config.storage import ROOT_DATA_FOLDER
from lib.tools.raster.services.clip import check_intersection, clip
from lib.tools.raster.services.translate import translate
from services.vector.io import read_geojson
from tasks.raster_processor import RasterProcessor
import geopandas as gpd


class ClipRasterProcessor(RasterProcessor):

    def __init__(self, task_id: int, token: str, group: str, target_id: int, payload: dict):
        super().__init__(task_id, token, group, target_id, payload)
        self.input_path = ROOT_DATA_FOLDER + payload.get('input_path')
        self.params = payload.get('params') or {}
        self.level = int(self.params.get('level')
                         ) if self.params.get('level') else 2

    def run_task(self):
        tmp_path = f'{self.temp_dir}/image.tif'
        mask = self.params.get('mask')
        mask_path = f'{self.temp_dir}/mask.geojson'
        with open(mask_path, 'w') as f:
            json.dump(mask, f)
        mask_gdf = read_geojson(mask_path)
        union_mask = json.loads(gpd.GeoSeries(mask_gdf.unary_union).to_json())
        union_mask_path = f"{self.temp_dir}/union_mask.json"
        print(union_mask)
        with open(union_mask_path, 'w') as f:
            json.dump(union_mask, f)
        final_mask = {
            "type": "FeatureCollection",
            "features":  check_intersection(self.input_path, union_mask_path)
        }

        final_mask_path = f"{self.temp_dir}/final_mask.json"
        with open(final_mask_path, 'w') as f:
            json.dump(final_mask, f)

        clip(self.input_path, final_mask_path, tmp_path)
        translate(tmp_path, self.output_path)
