import requests

import app.utils.broadcast as socket
from config.default import HOSTED_ENDPOINT


def change_task_status(task_id, status, error=None):
    payload = {
        'status': status,
    }
    if error:
        payload['error'] = str(error)
    url = '{}/internal/tasks/{}'.format(HOSTED_ENDPOINT, task_id)
    requests.put(url, json=payload)

def update_task_process(task_id, percentage):
    data = {
        'percentage': percentage
    }
    socket.task_proccessing_percent(task_id, data)

def get_task_args_from_payload(payload):
    return payload['user_id'], payload['wks_id'], payload['created_by']