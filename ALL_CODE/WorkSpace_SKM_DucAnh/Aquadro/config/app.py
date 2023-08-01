from config import env

DB_URL = env.str('DB_URL', '')
PLANET_FOLDER = env.str('PLANET_FOLDER', '')
AWS_ACCESS_KEY_ID = env.str('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = env.str('AWS_SECRET_ACCESS_KEY')
AWS_REQUEST_PAYER = env.str('AWS_REQUEST_PAYER', 'requester')

APP_URL = env.str('APP_URL', 'localhost:8000')
PLANET_KEY = env.str('PLANET_KEY', '7b2f361108194655b90bec2eeea0fb77')
DOCKER_REGISTRY = env.str('DOCKER_REGISTRY', 'registry.eofactory.ai:5000')
