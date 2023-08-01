
from abc import abstractmethod
from uuid import uuid4
from tasks import TaskExecutor
from services.vector.postgis import convert_to_postgis
from geopandas import GeoDataFrame
from services.db import engine


class VectorProcessor(TaskExecutor):
    output_gdf: GeoDataFrame

    def __init__(self, task_id: int, token, group: str, target_id: int, payload: dict):
        super().__init__(task_id, token, group, target_id, payload)
        self.output_type = 'shp'
        self.output_table_schema = 'personal_data'
        self.output_table_name = self.file_id

    @property
    def tile_url(self):
        return f'/tile/{self.group}/{self.target_id}/{self.output_table_schema}.{self.file_id}'+'/{z}/{x}/{y}.pbf'+f'?v={uuid4().hex}'

    @property
    def output_path(self):
        return f'{self.output_dir}/{self.file_id}.{self.output_type}'

    @abstractmethod
    def run_task(self):
        pass

    def write_file(self):
        self.output_gdf.to_file(self.output_path)

    def complete_task(self):
        self.write_file()
        self.convert_to_postgis()

    def convert_to_postgis(self):
        convert_to_postgis(self.output_gdf, self.output_table_name,
                           self.output_table_schema)

    def calculate_bbox(self):
        with engine.connect() as con:
            return list(con.execute(f'select\
                min(ST_XMin(ST_Transform(geometry, 4326))) as minx,\
                min(ST_YMin(ST_Transform(geometry, 4326))) as miny,\
                max(ST_XMax(ST_Transform(geometry, 4326))) as maxx,\
                max(ST_YMax(ST_Transform(geometry, 4326))) as maxy\
                from "{self.output_table_schema}"."{self.output_table_name}"').first())

    def on_success(self):
        self.store_result({
            "tile_url": self.tile_url,
            'path': self.relative_output_path,
            'crs': self.output_gdf.crs.to_string(),
            "size": self.get_file_size(self.output_path),
            'bbox': self.calculate_bbox(),
            'vector_type': self.output_gdf.geom_type.tolist()[0] if self.output_gdf.geom_type.size else None,
            'file_id': self.file_id
        })
