from app.job_queue.image_processors import ImageProcessor


class ZipImageProcessor(ImageProcessor):

    def __init__(self, user_id, workspace_id, created_by, **args):
        super().__init__(user_id, workspace_id, created_by, **args)

    def run_task(self):
        pass
