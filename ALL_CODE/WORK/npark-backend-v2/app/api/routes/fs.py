from flask import after_this_request

from app.controllers.fs_controller import FsController


def fs_routes_register(app, api):
    controller = FsController()

    @app.route('/pyapi/fs/download', methods=['GET'])
    def fs_download():
        return controller.download()

    @app.route('/pyapi/fs/download_report', methods=['GET'])
    def fs_download_report():
        return controller.download_report()
