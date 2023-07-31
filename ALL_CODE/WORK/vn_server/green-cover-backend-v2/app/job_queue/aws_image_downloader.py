from app.job_queue.image_downloader import ImageDownloader


class AWSImageDownloader(ImageDownloader):
    out_path = None

    def __init__(self, user_id, workspace_id, created_by, **args):
        super().__init__(user_id, workspace_id, created_by, **args)

    def prepare(self):
        pass

    def run_task(self):
        pass
