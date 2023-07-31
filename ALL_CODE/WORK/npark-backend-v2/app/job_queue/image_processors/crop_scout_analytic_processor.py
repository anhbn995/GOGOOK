from app.job_queue.task_executor import TaskExecutor


class CropScoutAnalyticProcessor(TaskExecutor):
    image_id = None

    def __init__(self, user_id, workspace_id, created_by, **args):
        super().__init__(user_id, workspace_id, created_by, **args)

    def run_task(self):
        pass

    def on_success(self):
        pass
