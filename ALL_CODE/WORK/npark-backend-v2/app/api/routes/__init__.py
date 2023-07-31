"""

"""
from flask_restful import Api
from flask_cors import CORS
from app.api.routes.imagery import imagery_routes_register
from app.api.routes.vector import vector_routes_register
from app.api.routes.fs import fs_routes_register
from .result import result_routes_register
from .aoi import aoi_routes_register
from .image_discovery import image_discovery_routes_register
from .internal.imageries import internal_imagery_routes_register
from .internal.fs import internal_fs_routes_register
from .internal.jobs import internal_job_routes_register


def register(app):
    """
    :param app:
    :return:
    """
    CORS(app)
    api = Api(app)
    fs_routes_register(app, api)
    imagery_routes_register(app, api)
    vector_routes_register(app, api)
    result_routes_register(app, api)
    aoi_routes_register(app, api)
    image_discovery_routes_register(app, api)
    internal_imagery_routes_register(app)
    internal_fs_routes_register(app)
    internal_job_routes_register(app)

