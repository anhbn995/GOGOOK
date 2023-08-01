import app.controllers.public_controller as pc


def public_routes_register(app):
    @app.route('/py/public/search', methods=['POST'])
    def search_images():
        return pc.search_images()

    @app.route('/py/public/statistics', methods=['POST'])
    def calculate_statistics():
        return pc.calculate_statistics()

    @app.route('/py/public/download', methods=['GET'])
    def download_image():
        return pc.download_image()

    @app.route('/py/public/zoning', methods=['GET'])
    def zoning():
        return pc.zoning()
