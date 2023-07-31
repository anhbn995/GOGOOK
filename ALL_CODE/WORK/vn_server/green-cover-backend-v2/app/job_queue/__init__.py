import json
import os
import time
import signal

from redis import Redis
from threading import Thread
from rq import Queue, Worker
from rq.job import Job
from enum import Enum

from app.utils.response import success
from app.job_queue.utils import get_task_args_from_payload
from config.default import REDIS_QUEUE_URL, REDIS_QUEUE_PORT

class QueueType(Enum):
    TRAINING = 'train'
    GPU_TF241 = 'gpu_tf241'
    MODEL_PREDICT = 'model_predict'
    PREDICTING = 'predict'
    UTILITY = 'utility'
    UTILITY_GPU = 'utility_gpu'
    UTILITY_TRICK = 'utility_trick'
    UTILITY_GPU_TF2 = 'utility_gpu_tf2'
    UTILITY_HIGH_MEMORY = 'utility_high_memory'
    UTILITY_GXL = 'utility_gxl'
    UTILITY_FME = 'utility_fme'

KILL_KEY = "rq:jobs:kill"

class KillJob(Job):
    def kill(self):
        """ Force kills the current job causing it to fail """
        if self.is_started:
            self.connection.sadd(KILL_KEY, self.get_id())

    def _execute(self):
        def check_kill(conn, id, interval=1):
            while True:
                res = conn.srem(KILL_KEY, id)
                if res > 0:
                    os.kill(os.getpid(), signal.SIGKILL)
                time.sleep(interval)

        t = Thread(target=check_kill, args=(self.connection, self.get_id()))
        t.start()

        return super()._execute()

class KillQueue(Queue):
    job_class = KillJob

class KillWorker(Worker):
    queue_class = KillQueue
    job_class = KillJob

_connection = Redis(host=REDIS_QUEUE_URL, port=REDIS_QUEUE_PORT, db=0)

train_task_queue = KillQueue(QueueType.TRAINING.value, connection=_connection)
tf241_task_queue = KillQueue(QueueType.GPU_TF241.value, connection=_connection)
predict_task_queue = KillQueue(QueueType.PREDICTING.value, connection=_connection)
model_predict_task_queue = KillQueue(QueueType.MODEL_PREDICT.value, connection=_connection)
utility_task_queue = KillQueue(QueueType.UTILITY.value, connection=_connection)
utility_trick_task_queue = KillQueue(QueueType.UTILITY_TRICK.value, connection=_connection)
utility_gpu_task_queue = KillQueue(QueueType.UTILITY_GPU.value, connection=_connection)
utility_gpu_tf2_task_queue = KillQueue(QueueType.UTILITY_GPU_TF2.value, connection=_connection)
utility_high_memory_queue = KillQueue(QueueType.UTILITY_HIGH_MEMORY.value, connection=_connection)
utility_gxl_task_queue = KillQueue(QueueType.UTILITY_GXL.value, connection=_connection)
utility_fme_task_queue = KillQueue(QueueType.UTILITY_FME.value, connection=_connection)


def get_queue_by_name(name):
    if name == QueueType.UTILITY.value:
        return utility_task_queue
    if name == QueueType.PREDICTING.value:
        return predict_task_queue
    if name == QueueType.MODEL_PREDICT.value:
        return model_predict_task_queue
    if name == QueueType.UTILITY_GPU.value:
        return utility_gpu_task_queue
    if name == QueueType.UTILITY_GPU_TF2.value:
        return utility_gpu_tf2_task_queue
    if name == QueueType.UTILITY_TRICK.value:
        return utility_trick_task_queue
    if name == QueueType.UTILITY_HIGH_MEMORY.value:
        return utility_high_memory_queue
    if name == QueueType.UTILITY_GXL.value:
        return utility_gxl_task_queue
    if name == QueueType.UTILITY_FME.value:
        return utility_fme_task_queue
    return train_task_queue

def get_queue_by_type(type):
    if type == QueueType.UTILITY:
        return utility_task_queue
    if type == QueueType.PREDICTING:
        return predict_task_queue
    if type == QueueType.MODEL_PREDICT:
        return model_predict_task_queue
    if type == QueueType.UTILITY_GPU:
        return utility_gpu_task_queue
    if type == QueueType.UTILITY_GPU_TF2:
        return utility_gpu_tf2_task_queue
    if type == QueueType.UTILITY_TRICK:
        return utility_trick_task_queue
    if type == QueueType.UTILITY_HIGH_MEMORY:
        return utility_high_memory_queue
    if type == QueueType.UTILITY_GXL:
        return utility_gxl_task_queue
    if type == QueueType.UTILITY_FME:
        return utility_fme_task_queue
    return train_task_queue

def job_wrapper(queue_type: QueueType, timeout='6h'):
    def decorator(function):
        def wrapper(*args, **kwargs):
            executor, request_data, payload_mapping = function(*args, **kwargs)
            payload = json.loads(request_data)
            job_kwargs = {
                'task_id': payload.get('task_id'),
                'name': payload.get('name'),
                'folder_id': payload.get('folder_id')
            }
            for target_key, origin_key in payload_mapping.items():
                job_kwargs[target_key] = payload.get(origin_key)
            queue = get_queue_by_type(queue_type)
            job = queue.enqueue(
                executor.execute,
                args=get_task_args_from_payload(payload),
                kwargs=job_kwargs,
                job_timeout=timeout
            )
            return success({
                'job_id': job.id,
                'queue_name': queue_type.value
            })
        return wrapper
    return decorator
