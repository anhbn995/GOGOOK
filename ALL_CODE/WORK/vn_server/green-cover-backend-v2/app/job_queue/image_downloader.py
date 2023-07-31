from app.job_queue.task_executor import TaskExecutor
from abc import abstractmethod


class ImageDownloader(TaskExecutor):

    def __init__(self, user_id, workspace_id, created_by, **args):
        super().__init__(user_id, workspace_id, created_by, **args)

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def run_task(self):
        pass

    def on_start(self):
        pass

    def on_success(self):
        pass

    def on_failed(self, err):
        print(err)
