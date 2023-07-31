from __future__ import print_function

import sys

from flask_script import Manager

from .shell import Shell
from .server import Server
from .database import db, manager as database_manager

from app.app import create_http_app
from config import to_envvar


def _create_app(config):
    if not to_envvar(config):
        print('Config file "{}" not found.'.format(config))
        sys.exit(1)

    app = create_http_app()
    return app


manager = Manager(_create_app)
manager.add_option("-c", "--config", dest="config", required=False)
manager.add_command("shell", Shell())
manager.add_command("runserver", Server(host="0.0.0.0", port='5003'))

manager.add_command("db", database_manager)
