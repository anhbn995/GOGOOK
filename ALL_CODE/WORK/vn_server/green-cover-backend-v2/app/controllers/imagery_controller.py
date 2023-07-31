import json
import logging
from flask import request
from flask_restful import Resource
from app.utils.authentication import auth
from app.services.imagery_service import ImageryService
from app.utils.response import success, error

logger = logging.getLogger('geoai')


class ImageryController(Resource):
    def __init__(self):
        self._service = ImageryService()

    @auth
    def clip_image_without_task(self, wks_id, img_id, **kwargs):
        user_id = kwargs.get('user_id')
        payload = json.loads(request.data)
        aois = payload.get('aois')
        res = self._service.clip_image_without_task(user_id, wks_id, img_id, aois)
        return res

    @auth
    def inspect_pixel_timeseries(self, wks_id, image_id, **kwargs):
        payload = request.args
        lat = payload.get('lat')
        lng = payload.get('lng')
        res = self._service.inspect_pixel_timeseries(wks_id, image_id, lat, lng)
        return res

    @auth
    def image_histogram_calculate(self, wks_id, image_id, **kwargs):
        res = self._service.image_histogram_calculate(wks_id, image_id)
        return res

    def download_image_client(self, user_id, wks_id, img_id):
        payload = request.args
        name = payload.get('name')
        type = payload.get('type')

        res = self._service.send_image_to_client(user_id, wks_id, img_id, name, type)
        return res

    def update_cut_line_prepare_mosaic(self, wks_id, prepare_id):
        payload = json.loads(request.data)
        if not payload.get('geojson'):
            raise Exception('Field geojson is required')
        if not payload.get('path'):
            raise Exception('Field path is required')
        res = self._service.update_cut_line(wks_id, prepare_id, payload)
        return res

    def get_unique_value_pixel(self, user_id, wks_id, img_id):
        payload = request.args
        band = 1
        if payload.get('band'):
            band = int(payload.get('band')) + 1
        res = self._service.get_unique_value_pixel(user_id, wks_id, img_id, band)
        if res.shape[0] > 300:
            return error("Too much values. Maximum value is 300")

        return success((res.tolist()))

    def get_color_table(self, year, month, img_id):
        payload = request.args
        band = 1
        if payload.get('band'):
            band = int(payload.get('band')) + 1
        res = self._service.get_color_table(year, month, img_id, band)
        return success(res)

    def get_range_value_reclassification(self, user_id, wks_id, img_id):
        payload = request.args

        band = 1
        if payload.get('band'):
            band = int(payload.get('band')) + 1

        type_range_value = "quantile"
        if payload.get('type'):
            type_range_value = payload.get('type')

        number_class = 1
        if payload.get('number_class'):
            number_class = int(payload.get('number_class'))

        res = self._service.get_range_value_reclassification(user_id, wks_id, img_id,band, type_range_value, number_class)

        return success(res)

    def store_metadata(self,user_id, wks_id, img_id):
        payload = json.loads(request.data)
        res = self._service.store_metadata(user_id, wks_id, img_id,payload.get('properties'))
        print(res)
        return success(res)
