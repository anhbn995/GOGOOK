import sys
sys.path.insert(0, '/home/geoai/npark-backend-v2')

from app import app
application = app.create_http_app("production")
