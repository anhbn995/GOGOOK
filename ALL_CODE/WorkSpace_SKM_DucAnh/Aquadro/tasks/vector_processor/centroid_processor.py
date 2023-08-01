from services.vector.postgis import read_postgis
from tasks.vector_processor import VectorProcessor


class CentroidProcessor(VectorProcessor):
    def __init__(self, task_id: int, token: str, group: str, target_id: int, payload: dict):
        super().__init__(task_id, token, group, target_id, payload)
        self.input_table_name = payload.get('input_table_name')
        self.input_table_schema = payload.get('input_table_schema')

    def run_task(self):
        self.output_gdf = read_postgis(
            f'SELECT * FROM "{self.input_table_schema}".{self.input_table_name}')
        self.output_gdf.geometry = self.output_gdf['geometry'].centroid
