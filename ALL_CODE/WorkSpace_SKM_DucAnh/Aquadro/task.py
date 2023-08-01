from celery.worker import state
from celery.utils.collections import LimitedSet
import time
from celery import Celery
from config.celery import CELERY_RESULT_BACKEND, CELERY_BROKER_URL
import os
import glob
from celery.signals import task_prerun, task_postrun, celeryd_after_setup
import logging
from config.storage import ROOT_DATA_FOLDER
LOG_FOLDER = f'{ROOT_DATA_FOLDER}/logs'


@celeryd_after_setup.connect
def setup_direct_queue(sender, instance, **kwargs):
    os.makedirs(LOG_FOLDER, exist_ok=True)


@task_prerun.connect
def overload_task_logger(task_id, task, args, **kwargs):
    import logging
    logger = logging.getLogger("celery")
    log_path = f'{ROOT_DATA_FOLDER}/logs/{task_id}.log'
    if os.path.exists(log_path):
        os.remove(log_path)
    fh = logging.FileHandler(log_path)
    logger.addHandler(fh)


@task_postrun.connect
def cleanup_logger(task_id, task, args, **kwargs):
    logger = logging.getLogger("celery")
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == f'{LOG_FOLDER}/{task_id}.log':
            logger.removeHandler(handler)


def detect_tasks():
    root_folder = os.path.dirname(os.path.abspath(__file__))+'/'
    tasks_folder = root_folder + 'tasks'
    files = glob.glob(f"{tasks_folder}/**/*.py", recursive=True)
    print([file.replace(root_folder, '').replace('.py', '').replace('/', '.') for file in files])
    return [file.replace(root_folder, '').replace('.py', '').replace('/', '.') for file in files]

celery = Celery(
    "mosaic",
    backend=CELERY_RESULT_BACKEND,
    broker=CELERY_BROKER_URL,
    include=detect_tasks()
)

celery.autodiscover_tasks(detect_tasks(), force=True)
celery.config_from_object('celeryconfig')

state.revoked = LimitedSet(maxlen=0, expires=1)
if __name__ == '__main__':
    celery.worker_main(['-A', 'task', 'worker', '--loglevel=INFO'])
