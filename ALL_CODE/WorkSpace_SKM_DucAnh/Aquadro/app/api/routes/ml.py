import app.controllers.ml_controller as tc


def ml_routes_register(app):
    @app.route('/py/public/ml/detect-crop', methods=['POST'])
    def detect_crop():
        return tc.detect_crop()
