from datetime import datetime
import json
import shutil
from config.storage import ROOT_DATA_FOLDER


from tasks.raster_processor import RasterProcessor
from lib.tools.raster.services.translate import translate
from sqlalchemy import column, table
from services.db import engine

class UploadRasterProcessor(RasterProcessor):
    def __init__(self, task_id: int, token: str, group: str, target_id: int, payload: dict):
        super().__init__(task_id, token, group, target_id, payload)
        self.path = ROOT_DATA_FOLDER + payload.get('path')
        self.info = payload.get('info') or {}
        self.table_name = payload.get('table_name')
        self.table_schema = payload.get('table_schema') or 'public'

    def run_task(self):
        tmp_path = f'{self.temp_dir}/image.tif'
        shutil.copyfile(self.path, tmp_path)
        translate(tmp_path, self.output_path)

    def on_success(self):
        created_at = updated_at = datetime.now()
        payload = {
            **self.info,
            "tile_url": self.tile_url,
            "metadata": json.dumps(self.metadata),
            "path": self.relative_output_path,
            'created_at': created_at,
            'updated_at': updated_at,
            'geometry': ST_SetSRID(ST_GeomFromGeoJSON(json.dumps(self.metadata['geometry'])), int(self.metadata['crs'].split(':')[1])),
            'thumbnail': self.thumbnail,
            'bbox': json.dumps(self.metadata['bbox'])
        }
        with engine.connect() as con:
            res = con.execute((
                table(self.table_name, *map(lambda key: column(key), payload.keys()), schema=self.table_schema
                      ).insert().values(**payload).returning(column("id"))
            ))
            self.feature_id = res.first()[0]
        super().on_success()
