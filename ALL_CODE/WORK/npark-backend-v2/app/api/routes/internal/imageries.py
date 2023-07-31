from app.controllers.internals import ic


def internal_imagery_routes_register(app):
    @app.route('/internals/imageries/download', methods=['GET', 'POST'])
    def down_from_source():
        return ic.download_image_from_source()

    @app.route('/internals/imageries/aws', methods=['POST'])
    def internal_aws_download():
        return ic.aws_download()

    @app.route('/internals/imageries/driver', methods=['POST'])
    def internal_driver_download():
        return ic.driver_download()

    @app.route('/pyapi/internals/imageries/zip_image', methods=['POST'])
    def internal_zip_image():
        return ic.zip_image()

    @app.route('/pyapi/internals/vectors/zip_vector', methods=['POST'])
    def internal_zip_vector():
        return ic.zip_vector()

    @app.route('/pyapi/internals/imageries/inspects', methods=['POST'])
    def internal_image_inspects():
        return ic.image_inspects()

    @app.route('/pyapi/internals/green_cover_change', methods=['POST'])
    def internal_green_cover_change():
        return ic.green_cover_change()
