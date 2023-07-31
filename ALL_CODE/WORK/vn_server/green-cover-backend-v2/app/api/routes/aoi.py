from app.controllers.aoi_controller import AOIController


def aoi_routes_register(app, api):
    controller = AOIController()

    @app.route('/public/geojson/convert', methods=['POST'])
    def geojson_convert():
        return controller.convert_to_geojson()

    @app.route('/public/geojson/convert_from_path', methods=['GET'])
    def convert_to_geojson_from_path():
        return controller.convert_to_geojson_from_path()

    @app.route('/internals/geojson/convert_crs', methods=['POST'])
    def geojson_convert_crs():
        return controller.geojson_convert_crs()

    @app.route('/internals/geojson/save_geojson', methods=['POST'])
    def save_geojson():
        return controller.save_geojson()

    @app.route('/internals/print/config_yaml', methods=['POST'])
    def create_yaml():
        return controller.create_yaml()

    @app.route('/public/single_color_image/<color>', methods=['GET'])
    def create_single_color_image(color):
        return controller.create_single_color_image(color)

    @app.route('/public/calculate_image', methods=['GET'])
    def calculate_image():
        return controller.calculate_image()
