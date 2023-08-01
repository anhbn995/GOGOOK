import app.controllers.convert_controller as tc


def utils_routes_register(app):
    @app.route('/py/public/read_kml', methods=['POST'])
    def read_kml():
        return tc.read_kml()

