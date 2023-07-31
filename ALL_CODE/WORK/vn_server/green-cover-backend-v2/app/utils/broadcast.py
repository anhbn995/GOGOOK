import requests
from threading import Thread
import config.default as env

BROADCAST_SERVER = env.GEOAI_BROADCAST_ENDPOINT
app_id = env.GEOAI_BROADCAST_APP_ID
key = env.GEOAI_BROADCAST_KEY
URL = "{}/apps/{}/events?auth_key={}".format(BROADCAST_SERVER, app_id, key)


def task_change_status(task_id, data):
    try:
        payload = {
            "channel": "private-task.{}".format(task_id),
            "name": "App\\Events\\TaskStatusChange",
            "data": data
        }
        thread = Thread(target=request, args=(URL, payload))
        thread.run()
    except Exception as e:
        print(e)
    return
    

def task_proccessing_percent(task_id, data):
    pass
    try:
        payload = {
            "channel": "private-task.{}".format(task_id),
            "name": "App\\Events\\TaskProccessingPercent",
            "data": data
        }
        thread = Thread(target=request, args=(URL, payload))
        thread.start()
    except Exception as e:
        print(e)
    return


def request(url, payload):
    requests.post(url=url, json=payload)
