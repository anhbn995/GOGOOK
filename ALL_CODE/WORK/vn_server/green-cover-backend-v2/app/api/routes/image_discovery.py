from app.controllers.image_discovery_controller import ImageDiscoveryController


def image_discovery_routes_register(app, api):
    controller = ImageDiscoveryController()

    @app.route('/imagery/discovery', methods=['POST'])
    def image_discovery():
        return controller.search()
