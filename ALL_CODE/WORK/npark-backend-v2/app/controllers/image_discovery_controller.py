import json

from flask import request
from flask_restful import Resource

from app.services.image_discovery import ImageDiscoveryFactory

class ImageDiscoveryController(Resource):
    def __init__(self):
        self._factory = ImageDiscoveryFactory()

    def search(self):
        payload = json.loads(request.data)
        return self._factory.search(**payload)
