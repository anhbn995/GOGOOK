"""
    :author: Nghi Pham
"""
import json
from flask import request
from app.utils.response import success
from app.job_queue import get_queue_by_name


def kill():
    """

    :return:
    """
    payload = json.loads(request.data)
    queue_name = payload.get('queue_name')
    job_id = payload.get('job_id')
    try:
        queue = get_queue_by_name(queue_name)
        job = queue.fetch_job(job_id)
        job.kill()
    except Exception as error:
        raise error
    return success('Successful')

def delete():
    """

    :return:
    """
    payload = json.loads(request.data)
    queue_name = payload.get('queue_name')
    job_id = payload.get('job_id')
    try:
        queue = get_queue_by_name(queue_name)
        job = queue.fetch_job(job_id)
        job.delete()
    except Exception as error:
        print(error)
    return success('Successful')

def requeue():
    """
    :parameter:
    queue_name (string): queue identify
    job_id (string): job identify that we want to requeue
    :return:
    """
    payload = json.loads(request.data)
    queue_name = payload.get('queue_name')
    job_id = payload.get('job_id')
    try:
        queue = get_queue_by_name(queue_name)
        job = queue.fetch_job(job_id)
        registry = queue.failed_job_registry
        registry.requeue(job)
    except Exception as error:
        raise error
    return success('Successful')
