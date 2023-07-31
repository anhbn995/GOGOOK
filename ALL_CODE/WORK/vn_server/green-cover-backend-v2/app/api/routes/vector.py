from app.controllers.vector_controller import VectorController


def vector_routes_register(app, api):
    controller = VectorController()

    @app.route('/<user_id>/<wks_id>/<type_download>/<vector_id>/download', methods=['GET'])
    def download_vector(user_id, wks_id, type_download, vector_id):
        return controller.download_vector(user_id, wks_id, type_download, vector_id)
