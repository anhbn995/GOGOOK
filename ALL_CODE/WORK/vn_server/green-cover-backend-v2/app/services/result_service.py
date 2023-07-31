import os
import uuid

from app.utils.response import not_found

from app.models.result import Result
from flask import send_from_directory


class ResultService:

    def download_raw(self, result_id):
        result = Result.query().filter_by(id=result_id).first()
        data_dir = result.path
        print(data_dir)
        if not os.path.exists(f'{data_dir}{uuid.UUID(str(result_id)).hex}.tif'):
            return not_found('File not found')
        return send_from_directory(data_dir, "{}.tif".format(uuid.UUID(str(result_id)).hex))
