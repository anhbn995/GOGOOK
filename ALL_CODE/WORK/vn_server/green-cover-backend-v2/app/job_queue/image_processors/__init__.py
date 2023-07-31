from abc import abstractmethod

from app.job_queue.task_executor import TaskExecutor


class ImageProcessor(TaskExecutor):

    def __init__(self, user_id, workspace_id, created_by, **kwargs):
        super().__init__(user_id, workspace_id, created_by, **kwargs)

    def prepare(self):
        pass

    @abstractmethod
    def run_task(self):
        pass

    def on_start(self):
        pass

    def complete_task(self):
        pass

    def on_success(self):
        pass

    def on_failed(self, err):
        pass




