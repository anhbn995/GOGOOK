import sys
sys.path.insert(0, '/home/geoai/api.eofactory.ai')

from app import app
application = app.create_http_app("production")

