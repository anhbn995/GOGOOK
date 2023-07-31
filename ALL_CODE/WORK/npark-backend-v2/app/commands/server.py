from flask_script import Server as BaseServer
from app.app import create_http_app


class Server(BaseServer):
    def handle(self, app, *args, **kw):
        app = create_http_app()
        super(Server, self).handle(app, *args, **kw)