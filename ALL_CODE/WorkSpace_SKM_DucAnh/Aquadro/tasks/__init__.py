

from abc import ABC, abstractmethod
import os
import shutil
import uuid

import requests
from config.storage import DATA_EXPLORER_URL
from config.storage import ROOT_DATA_FOLDER
from services.task import update_task
from services.rq import queue


class TaskExecutor(ABC):

    def __init__(self, task_id: int, token: str, group: str, target_id: int, payload: dict):
        self.task_id = task_id
        self.payload = payload
        self.group = group
        self.target_id = target_id
        self.file_id = uuid.uuid4().hex
        self.temp_dir = f'{ROOT_DATA_FOLDER}/tmp/{uuid.uuid4().hex}'
        self.token = token

    @property
    def logs_dir(self):
        return f'{ROOT_DATA_FOLDER}/logs'

    @property
    def logs_path(self):
        return f'{ROOT_DATA_FOLDER}/logs/{self.task_id}.txt'

    @property
    def output_dir(self):
        return f'{ROOT_DATA_FOLDER}/{self.group}' + (f'/{self.target_id}' if self.target_id else '')

    @property
    def relative_output_path(self) -> str:
        return self.output_path.replace(ROOT_DATA_FOLDER, '')

    @property
    @abstractmethod
    def output_path(self) -> str:
        pass

    @property
    def output_size(self):
        return os.path.getsize(self.output_path)

    def prepare(self):
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def run_task(self):
        pass

    def complete_task(self):
        pass

    @abstractmethod
    def on_success(self):
        pass

    def store_result(self, payload):
        payload = {**payload, 'task_id':self.task_id}
        url = '{}/api/internal/{}/{}'.format(
            DATA_EXPLORER_URL, self.group, self.target_id)
        r = requests.put(url, json=payload, headers={
            'Authorization': 'Bearer {}'.format(self.token)}, timeout=600)
        if not r.ok:
            print(r.json())
            raise Exception('An error has occured when updating dataset')

    def update_task(self, status, payload):
        update_task(self.task_id, status, payload)

    def handle_exception(self, exception):
        pass

    @classmethod
    def enqueue(cls, task_id, token, group, target_id, payload):
        job = queue.enqueue(cls.execute, args=(
            task_id, token, group, target_id, payload), job_timeout=3600)
        return {
            "message": 'Task created successfully',
            "job_id": job.id
        }

    @classmethod
    def execute(cls, task_id, token, group, target_id, payload):
        try:
            instance = cls(task_id, token, group, target_id, payload)
            instance.prepare()
            instance.run_task()
            instance.complete_task()
            instance.on_success()
        except Exception as exception:
            instance.handle_exception(exception)
            raise exception
        finally:
            pass
            # shutil.rmtree(instance.temp_dir)

    def get_file_size(self, path):
        return os.path.getsize(path)
