import app.controllers.tile_controller as tc


def tile_routes_register(app):
    @app.route('/py/api/sentinel2/<img_id>/statistic', methods=['GET'])
    def sentinel2_statistic(img_id):
        return tc.sentinel2_statistic(img_id)

    @app.route('/py/tiles/sentinel/<img_id>/<z>/<x>/<y>.png', methods=['GET'])
    def sentinel_serve_tile(img_id, z, x, y):
        return tc.sentinel_serve_tile(img_id, z, x, y)

    @app.route('/py/tiles/sentinel2/cloud/<img_id>/<z>/<x>/<y>.png', methods=['GET'])
    def sentinel2_cloud_tile(img_id, z, x, y):
        return tc.sentinel2_cloud_tile(img_id, z, x, y)

    @app.route('/py/tiles/planet/<img_id>/<z>/<x>/<y>.png', methods=['GET'])
    def planet_tile(img_id, z, x, y):
        return tc.planet_tile(img_id, z, x, y)

    @app.route('/py/tiles/planet/cloud/<img_id>/<z>/<x>/<y>.png', methods=['GET'])
    def planet_cloud_tile(img_id, z, x, y):
        return tc.planet_cloud_tile(img_id, z, x, y)