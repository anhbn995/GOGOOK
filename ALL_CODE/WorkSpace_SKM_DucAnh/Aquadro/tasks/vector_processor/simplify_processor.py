from services.vector.postgis import read_postgis
from tasks.vector_processor import VectorProcessor


class SimplifyProcessor(VectorProcessor):
    def __init__(self, task_id: int, token: str, group: str, target_id: int, payload: dict):
        super().__init__(task_id, token, group, target_id, payload)
        self.input_table_name = payload.get('input_table_name')
        self.input_table_schema = payload.get('input_table_schema')
        self.params = payload.get('params') or {}

    def run_task(self):
        self.output_gdf = read_postgis(
            f'SELECT * FROM "{self.input_table_schema}".{self.input_table_name}')
        options = {
            "tolerance": float(self.params.get('tolerance')) if self.params.get('tolerance') else 1.0,
        }
        self.output_gdf.geometry = self.output_gdf['geometry'].simplify(
            **options)
