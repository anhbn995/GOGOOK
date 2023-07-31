import logging
from flask import request
from flask_restful import Resource
from flask import send_from_directory
from app.models.workspace import get_workspace_dir
import zipfile
import os

logger = logging.getLogger('geoai')


class VectorController(Resource):

    def download_vector(self, user_id, wks_id, type_download, vector_id):
        payload = request.args
        name = payload.get('name')
        file_type = payload.get('file_type')
        dir_path = get_workspace_dir(user_id, wks_id)
        vector_path = f"{dir_path}/{type_download}"

        file_type_zips = ['shx', 'shp', 'dbf', 'prj', 'qix' 'kml', 'kmz']
        if not name:
            name = vector_id
        if not file_type or file_type == 'shp':
            file_type = 'zip'

        if file_type == 'zip':
            if not os.path.exists(f"{vector_path}/{vector_id}/{name}.{file_type}"):
                os.mkdir(f"{vector_path}/{vector_id}")
                with zipfile.ZipFile(f"{vector_path}/{vector_id}/{name}.{file_type}", 'w') as my_zip:
                    for file_type_zip in file_type_zips:
                        if os.path.exists(os.path.join(vector_path, f'{vector_id}.{file_type_zip}')):
                            my_zip.write(os.path.join(vector_path, f'{vector_id}.{file_type_zip}'),
                                         arcname=f'{name}.{file_type_zip}')

            return send_from_directory(f"{vector_path}/{vector_id}", f"{name}.{file_type}", as_attachment=True)
        else:
            return send_from_directory(f"{vector_path}", f"{vector_id}.{file_type}",
                                       attachment_filename=f"{name}.{file_type}", as_attachment=True)
