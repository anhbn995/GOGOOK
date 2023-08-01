import sys
sys.path.insert(0, '/app')

from app import app
application = app.create_http_app("production")