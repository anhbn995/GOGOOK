from glob import glob
import shutil
from zipfile import ZipFile
from config.storage import ROOT_DATA_FOLDER
from services.vector.dataframe import get_geodataframe
from services.vector.postgis import convert_to_postgis
from tasks.vector_processor import VectorProcessor


class UploadVectorProcessor(VectorProcessor):
    def __init__(self, task_id: int, token: str, group: str, target_id: int, payload: dict):
        super().__init__(task_id, token, group, target_id, payload)
        self.path = ROOT_DATA_FOLDER + payload.get('path')
        self.output_table_name = payload.get('table_name')
        self.output_table_schema = payload.get('table_schema')
        self.output_type = self.path.split('.')[-1]
        self.file_id = self.output_table_name
        self.fields = payload.get('fields')

    def run_task(self):
        self.output_gdf = get_geodataframe(self.path)

    def write_file(self):
        if self.output_type != 'shp':
            return shutil.copyfile(self.path, self.output_path)
        self.output_type = 'zip'
        with ZipFile(self.output_path, 'w') as zip:
            for f in glob(self.path.replace('shp', '*')):
                zip.write(f, f.split('/')[-1])

    def convert_to_postgis(self):
        convert_to_postgis(self.output_gdf, self.output_table_name,
                           self.output_table_schema, self.fields)
