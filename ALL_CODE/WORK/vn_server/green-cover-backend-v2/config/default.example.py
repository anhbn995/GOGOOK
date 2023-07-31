#
#  -*- coding: utf-8 -*-

import os.path

here = os.path.abspath(os.path.dirname(__file__))
base = os.path.abspath(os.path.join(here, ".."))

DEBUG_TB_INTERCEPT_REDIRECTS = False

DEBUG = True
SECRET_KEY = "geoai"

SMTP_SERVER = "localhost"
EMAIL_FROM = "supports@skymapglobal.com"
EMAIL_PASSWORD = "smgsupport@123"

"""
# Logging handlers (disabled in DEBUG mode)

file_handler = (
    'INFO', 'RotatingFileHandler',
    ('/home/turbo/GEOAI.log', 'a', 10000, 4))

LOGGING_HANDLERS = [file_handler]

SENTRY_DSN = 'https://foo:bar@sentry.io/appid'
"""

# This should probably be changed for a multi-threaded production server
CACHE_TYPE = "simple"

REDIS_QUEUE_URL = "skymapglobal.vn"
REDIS_QUEUE_PORT = 6379

GEOAI_GATEWAY_AUTH = "http://skymapglobal.vn:3001/me"
GEOAI_BROADCAST_ENDPOINT = "http://skymapglobal.vn:6001"
GEOAI_BROADCAST_APP_ID = "443a198077ff03e6"
GEOAI_BROADCAST_KEY = "611551368d382a4dbc3045d5e96d5c81"

SQLALCHEMY_DATABASE_URI = "postgres://postgres:admin_123@biztrekk.com:5432/geoai"
SQLALCHEMY_POOL_SIZE = 20
SQLALCHEMY_TRACK_MODIFICATIONS = False

ASSETS_LOAD_DIR = os.path.join(base, "geoai", "frontend", "static")

GEOAI_FILES_PATH = os.path.join(base, "htdocs", "files")
GEOAI_ELEVATION_PATH = os.path.join(base, "htdocs", "srtm")
GEOAI_MAPSERVER_PATH = os.path.join(base, "mapserver")

GEOAI_TEMPORARY_DIR = "/tmp"

TENSORRT_SERVER_URL = "biztrekk.com:8000"

ROOT_DATA_FOLDER = "."
AWS_DATA_FOLDER = "."

SNAP_OPT = '/home/boom/snap/bin/gpt'
XML_FILE_50m = '/home/boom/geoai_v2/geoai-backend-v2/config/xml_graph/graph_mlc_50m.xml'
XML_FILE_10m = '/home/boom/geoai_v2/geoai-backend-v2/config/xml_graph/sentinel_1_grd_10m_sigma0.xml'

PYTHON_ENV_PATH = "/home/boom/miniconda3/envs/api/bin/python"

EE_DOWNLOAD_ENDPOINT = "http://0.0.0.0:9999/download_ee"

GDAL_BIN = "/home/geoai/anaconda3/envs/geoai/bin/gdal_merge.py"

UPLOAD_FOLDER = "/storage/"

LIBRARY_FOLDER = "/library/"

HOSTED_ENDPOINT = "http://localhost:5001"

STRIPE_KEY = "sk_test_BxIKeaUJwOm2gBd9qo8ztDEu00M2A4W7Oy"

CUBE_HOST = "http://192.168.1.179:8000"

CLOUD_PATH = "/home/boom/Desktop/test/image-processing-master/cloudremoval"

AWS_FOLDER = "./AwsData"

COMMUNITY_FOLDER = "/community/"

X_RAPIDAPI_KEY = ""

MITTAL_EMAIL = "abhay.mittal@skymapglobal.com"

CROP_SCOUT_DATABASE = "postgresql://postgres:admin_123@192.168.1.119:5433/crop_scout"

AWS_BUCKET_NAME = "eofactory"

GEOAI_TEMPORARY_DIR = "/tmp"

PUBLIC_TILE_URL = "http://0.0.0.0:5002"

SENTINEL_API_USER=
SENTINEL_API_PASSWORD=
SENTINEL_API_URL="https://apihub.copernicus.eu/apihub/"

REGION= #SEA or EU