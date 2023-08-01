import requests
from config.storage import DATA_EXPLORER_URL


def update_task(task_id, token: str = '', payload: dict = {}, ):
    url = '{}/api/internal/tasks/{}'.format(DATA_EXPLORER_URL, task_id)
    requests.put(url, json=payload, headers={
        'Authorization': 'Bearer {}'.format(token)}, timeout=600)
