from services.vector.postgis import read_postgis
from tasks.vector_processor import VectorProcessor


class BufferProcessor(VectorProcessor):
    def __init__(self, task_id: int, token: str, group: str, target_id: int, payload: dict):
        super().__init__(task_id, token, group, target_id, payload)
        self.input_table_name = payload.get('input_table_name')
        self.input_table_schema = payload.get('input_table_schema')
        self.params = payload.get('params') or {}

    def run_task(self):
        self.output_gdf = read_postgis(
            f'SELECT * FROM "{self.input_table_schema}".{self.input_table_name}')
        options = {
            "distance": int(self.params.get('distance')),
            # default: 16
            "resolution": int(self.params.get('resolution')) if self.params.get('resolution') else 16,
            # default: None
            "quadsegs": int(self.params.get('quadsegs')) if self.params.get('quadsegs') else None,
            # round:1 , flat: 2, square: 3 default: round
            "cap_style": self.params.get('cap_style') or 1,
            # round:1, mitre:2, bevel:3 default: round
            "join_style": self.params.get('join_style') or 1,
            # default: 5.0
            "mitre_limit": float(self.params.get('mitre_limit')) if self.params.get('mitre_limit') else 5.0,
        }

        self.output_gdf.geometry = self.output_gdf['geometry'].buffer(
            **options)
