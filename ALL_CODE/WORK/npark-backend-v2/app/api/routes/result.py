from app.controllers.result_controller import ResultController


def result_routes_register(app, api):
    controller = ResultController()

    @app.route('/workspaces/<wks_id>/results/<result_id>/raw', methods=['GET'])
    def results_raw(wks_id, result_id):
        return controller.download_raw(wks_id, result_id)

    @app.route('/result_tile/<user_id>/<wks_id>/<result_id>/<z>/<x>/<y>', methods=['GET'])
    def result_tile(user_id, wks_id, result_id, z, x, y):
        return controller.tile(user_id, wks_id, result_id, z, x, y)
