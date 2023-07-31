DEBUG = True
SECRET_KEY = "geoai"

REDIS_QUEUE_URL = "localhost"
REDIS_QUEUE_PORT = 6379

GEOAI_GATEWAY_AUTH = "https://authtest.eofactory.ai:3443/me"
GEOAI_BROADCAST_ENDPOINT = "http://192.168.1.101:6001"
GEOAI_BROADCAST_APP_ID = "443a198077ff03e6"
GEOAI_BROADCAST_KEY = "611551368d382a4dbc3045d5e96d5c81"

SQLALCHEMY_POOL_SIZE = 20
SQLALCHEMY_TRACK_MODIFICATIONS = False

PUBLIC_TILE_URL = "http://0.0.0.0:5004"
SENTINEL_API_USER = "lehai.ha"
SENTINEL_API_PASSWORD = "DangKhoa@123"
AMQP_HOST = "192.168.4.100"
AMQP_PORT = 5672
AMQP_VHOST = "/eof"
AMQP_USERNAME = "eof_rq_worker"
AMQP_PASSWORD = "123"
SENTINEL_API_URL = "https://apihub.copernicus.eu/apihub/"

# config test sentinel2
ROOT_DATA_FOLDER = "/home/geoai/geoai_data_test2/data_npark_prod"
HOSTED_ENDPOINT = "https://nparkapi.eofactory.ai:3443/api/v1"
SQLALCHEMY_DATABASE_URI = "postgresql://postgres:admin_123@192.168.4.100/npark2_backup"
NPARK_DATA_FOLDER = '/home/geoai/geoai_data_test2/NPARK_DATA'
# confix production
# ROOT_DATA_FOLDER = "/home/geoai/geoai_data_test2/data_npark_service"
# HOSTED_ENDPOINT = "https://nparksapi.skymapdataservice.com/api/v1"
# SQLALCHEMY_DATABASE_URI = "postgresql://postgres:postgres@23.106.120.91/npark2"
