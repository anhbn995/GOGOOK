from flask import Flask
from app.api.cache import cache


class EOFTile(Flask):
    def __init__(self, name="eoftile", config_file=None, *args, **kw):
        # Create Flask instance
        super(EOFTile, self).__init__(name, *args, **kw)

    def add_logging_handlers(self):
        if self.debug:
            return

        import logging
        from logging import Formatter
        from logging.handlers import RotatingFileHandler

        # Set general log level
        self.logger.setLevel(logging.INFO)

        # Add log file handler (if configured)
        path = self.config.get("LOGFILE")
        if path:
            file_handler = RotatingFileHandler(path, "a", 10000, 4)
            file_handler.setLevel(logging.INFO)

            file_formatter = Formatter(
                "%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]"
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        self.logger.info("init app! done")


def create_app(*args, **kw):
    app = EOFTile(*args, **kw)
    from app.api.routes import register
    from app.api.errors import register as register_error_handlers
    app.app_context().push()
    register(app)
    register_error_handlers(app)
    app.config['SECRET_KEY'] = 'secret!'
    app.config['DEBUG'] = True
    cache.init_app(app)
    return app


def create_http_app(*args, **kw):
    app = create_app(*args, **kw)
    app.add_logging_handlers()
    return app
