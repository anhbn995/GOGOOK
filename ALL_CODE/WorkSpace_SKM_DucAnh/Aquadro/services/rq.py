from datetime import datetime
import traceback
from redis import Redis
from rq import Queue, Worker
from config.rq import REDIS_QUEUE_PORT, REDIS_QUEUE_HOST
from rq.job import Job
from rq.command import send_stop_job_command
from services.task import update_task
from config.enum import TaskStatus

redis = Redis(host=REDIS_QUEUE_HOST, port=REDIS_QUEUE_PORT)

queue = Queue(connection=redis)


class CustomWorker(Worker):
    def handle_exception(self, job, *exc_info):
        exc_string = ''.join(traceback.format_exception(*exc_info))
        update_task(job.args[0], job.args[1], {
            'status': TaskStatus.FAILED.value,
            'error': str(exc_info[1]) if str(exc_info[1]) else exc_string,
            'start_at': datetime.now().isoformat()
        })
        super().handle_exception(job, *exc_info)

    def execute_job(self, job, queue):
        update_task(job.args[0], job.args[1], {
            'status': TaskStatus.PROCESSING.value,
            'job_id': job.id,
            'start_at': datetime.now().isoformat()
        })
        super().execute_job(job, queue)

    def handle_job_success(self, job, queue, started_job_registry):
        update_task(job.args[0], job.args[1], {
            'status': TaskStatus.COMPLETED.value,
            'end_at': datetime.now().isoformat()
        })
        super().handle_job_success(job, queue, started_job_registry)


def cancel_job(job_id):
    send_stop_job_command(redis, job_id)
    return {
        'message': 'Job cancelled successfully',
        'job_id': job_id
    }


def requeue(job_id):
    job = Job.fetch(job_id, connection=redis)
    job.requeue()
    return {
        'message': 'Job requeued successfully',
        'job_id': job_id
    }
