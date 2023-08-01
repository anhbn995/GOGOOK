from config import env
from config.storage import ROOT_DATA_FOLDER
BACKEND_FOLDER = f'file://{ROOT_DATA_FOLDER}/results'
CELERY_RESULT_BACKEND = env.str('CELERY_RESULT_BACKEND', BACKEND_FOLDER)
CELERY_BROKER_URL = env.str('CELERY_BROKER_URL', 'amqp://guest:guest@localhost:5672//')