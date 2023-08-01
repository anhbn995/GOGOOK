import json
from services.vector.io import read_geojson
from services.vector.postgis import read_postgis
from tasks.vector_processor import VectorProcessor
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon


class ClipVectorProcessor(VectorProcessor):
    def __init__(self, task_id: int, token: str, group: str, target_id: int, payload: dict):
        super().__init__(task_id, token, group, target_id, payload)
        self.input_table_name = payload.get('input_table_name')
        self.input_table_schema = payload.get('input_table_schema')
        self.params = payload.get('params') or {}

    def remove_invalid_geometry(self, gdf):
        for ind, g in enumerate(gdf['geometry']):
            if not type(g) in [Polygon, MultiPolygon] or not g.is_valid or g.is_empty:
                gdf = gdf.drop(ind)
        return gdf

    def remove_empty_data(self, gdf):
        return gdf[~(gdf['geometry'].is_empty | gdf['geometry'].isnull())]

    def run_task(self):
        input_gdf = read_postgis(
            f'SELECT * FROM "{self.input_table_schema}".{self.input_table_name}')
        mask = self.params.get('mask')
        mask_path = f'{self.temp_dir}/mask.geojson'
        with open(mask_path, 'w') as f:
            json.dump(mask, f)
        mask_gdf = self.remove_invalid_geometry(
            read_geojson(mask_path)).to_crs(input_gdf.crs)

        if input_gdf['geometry'].count() == 0:
            raise Exception("Input is empty")
        self.output_gdf = gpd.clip(input_gdf, mask_gdf)
        if self.output_gdf.empty:
            raise Exception('Empty result!')
