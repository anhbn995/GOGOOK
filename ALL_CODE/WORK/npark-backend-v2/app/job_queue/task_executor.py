from abc import ABC, abstractmethod

from app.job_queue.utils import *


class TaskExecutor(ABC):
    
    task_id = None
    result_type = None

    def __init__(self, user_id, workspace_id, created_by, **args):
        pass

    @classmethod
    def execute(cls, user_id, workspace_id, created_by, **args):
        pass

    def complete_task(self):
        pass

    def prepare(self):
        pass 

    def on_start(self):
        pass

    def on_processing(self, percent):
        pass

    @abstractmethod
    def on_success(self):
        pass
        
    def on_failed(self, err):
        pass

    @abstractmethod
    def run_task(self):
        pass

