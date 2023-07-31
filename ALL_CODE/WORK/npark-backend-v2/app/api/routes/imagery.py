from app.controllers.imagery_controller import ImageryController

def imagery_routes_register(app, api):
    controller = ImageryController()

    @app.route('/workspaces/<wks_id>/imageries/<img_id>/clip_without_task', methods=['POST'])
    def clip_image_without_task(wks_id, img_id):
        return controller.clip_image_without_task(wks_id, img_id)

    @app.route('/workspaces/<wks_id>/imageries/<img_id>/inspect', methods=['GET'])
    def inspect(wks_id, img_id):
        return controller.inspect_pixel_timeseries(wks_id, img_id)

    @app.route('/workspaces/<wks_id>/imageries/<img_id>/histogram', methods=['GET'])
    def histogram(wks_id, img_id):
        return controller.image_histogram_calculate(wks_id, img_id)

    @app.route('/<user_id>/<wks_id>/imageries/<img_id>/download', methods=['GET'])
    def download(user_id, wks_id, img_id):
        return controller.download_image_client(user_id, wks_id, img_id)

    @app.route('/workspaces/<wks_id>/prepare_mosaics/<prepare_id>/cutline', methods=['PUT'])
    def update_cut_line_prepare_mosaic(wks_id, prepare_id):
        return controller.update_cut_line_prepare_mosaic(wks_id, prepare_id)

    @app.route('/user/<user_id>/workspaces/<wks_id>/imageries/<img_id>/unique_value', methods=['GET'])
    def get_unique_value_pixel(user_id,wks_id, img_id):
        return controller.get_unique_value_pixel(user_id,wks_id, img_id)

    @app.route('/<year>/<month>/imageries/<img_id>/color_table', methods=['GET'])
    def get_color_table(year, month, img_id):
        return controller.get_color_table(year, month, img_id)

    @app.route('/user/<user_id>/workspaces/<wks_id>/imageries/<img_id>/range_value_reclassification', methods=['GET'])
    def get_range_value_reclassification(user_id, wks_id, img_id):
        return controller.get_range_value_reclassification(user_id, wks_id, img_id)

    @app.route('/user/<user_id>/workspaces/<wks_id>/imageries/<img_id>/store_metadata', methods=['POST'])
    def store_metadata(user_id, wks_id, img_id):
        return controller.store_metadata(user_id, wks_id, img_id)

