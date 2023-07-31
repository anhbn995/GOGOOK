import logging

from flask_restful import Resource

from app.utils.authentication import auth
from app.services.result_service import ResultService


logger = logging.getLogger('geoai')


class ResultController(Resource):
    def __init__(self):
        self._service = ResultService()

    def download_raw(self, wks_id, result_id):
        res = self._service.download_raw(result_id)
        return res

    @auth
    def delete(self, wks_id, result_id, **kwargs):
        user_id = kwargs.get('user_id')
        res = self._service.destroy_result(user_id, result_id)
        return res
