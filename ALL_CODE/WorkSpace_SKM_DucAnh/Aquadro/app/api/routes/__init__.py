from flask_cors import CORS
from app.api.routes.tile import tile_routes_register
from app.api.routes.public import public_routes_register
from app.api.routes.ml import ml_routes_register
from app.api.routes.utils import utils_routes_register


def register(app):
    CORS(app)
    tile_routes_register(app)
    public_routes_register(app)
    ml_routes_register(app)
    utils_routes_register(app)

