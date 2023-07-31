import logging
from flask import request, send_from_directory
from flask_restful import Resource
import os
import uuid
import shutil
from flask import Response
from config.default import ROOT_DATA_FOLDER
from app.utils.path import make_temp_folder

logger = logging.getLogger('geoai')


class FsController(Resource):

    def download(self):
        payload = request.args
        file_path = f"{payload.get('path')}"
        file_name = payload.get('name')

        if not os.path.exists(file_path):
            file_path = f"{ROOT_DATA_FOLDER}{file_path}"

        if os.path.exists(file_path):
            file_name_original = os.path.basename(file_path)
            if os.path.isfile(file_path):
                if not file_name:
                    file_name = file_name_original
                else:
                    file_name = f"{file_name}.zip"
                dir_path = os.path.dirname(file_path)
            else:
                dir_path = f"{ROOT_DATA_FOLDER}/raw"
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                    
                if not os.path.exists(f"{dir_path}/{file_name_original}.zip"):
                    shutil.make_archive(f"{dir_path}/{file_name_original}", 'zip', file_path)
                file_name_original = f"{file_name_original}.zip"
                if not file_name:
                    file_name = f"{os.path.basename(file_path)}.zip"
                else:
                    file_name = f"{file_name}.zip"

            if not os.path.isfile(file_path):
                with open(os.path.join(dir_path, file_name_original), 'rb') as f:
                    data = f.readlines()
                import time
                time.sleep(2)
                # os.remove(os.path.join(dir_path, file_name_original))
                # return Response(data, headers={
                #     'Content-Type': 'application/zip',
                #     'Content-Disposition': 'attachment; filename=%s;' % f"{file_name}"
                # })
                return send_from_directory(dir_path, file_name_original,
                                           attachment_filename=file_name, as_attachment=True)
            else:
                return send_from_directory(dir_path, file_name_original,
                                           attachment_filename=file_name, as_attachment=True)

    def download_report(self):

        payload = request.args
        month = f"{payload.get('month')}"
        year = payload.get('year')
        file_name = f'Report {month}_{year}.pdf'

        dir_path = f"{ROOT_DATA_FOLDER}/data/reports"

        return send_from_directory(dir_path, file_name,
                                   attachment_filename=file_name, as_attachment=True)

