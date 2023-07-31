import os
import config

from flask import Flask


class GeoAI(Flask):
    def __init__(self, name="geoai", config_file=None, *args, **kw):
        # Create Flask instance
        super(GeoAI, self).__init__(name, *args, **kw)

        # Load default settings and from environment variable
        self.config.from_pyfile(config.DEFAULT_CONF_PATH)

        if "GEOAI_CONFIG" in os.environ:
            self.config.from_pyfile(os.environ["GEOAI_CONFIG"])

        if config_file:
            self.config.from_pyfile(config_file)

    def add_sqlalchemy(self):
        """ Create and configure SQLAlchemy extension """
        from app.database import db

        db.init_app(self)

    def add_cache(self):
        """ Create and attach Cache extension """
        from app.api.cache import cache

        cache.init_app(self)

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
    app = GeoAI(*args, **kw)
    app.add_sqlalchemy()
    from app.api.routes import register
    from app.api.errors import register as register_error_handlers
    register(app)
    # register_error_handlers(app)
    app.config['SECRET_KEY'] = 'secret!'
    return app

def create_http_app(*args, **kw):
    app = create_app(*args, **kw)
    app.add_logging_handlers()
    return app
