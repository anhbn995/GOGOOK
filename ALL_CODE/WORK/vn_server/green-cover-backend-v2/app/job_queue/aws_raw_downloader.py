from app.job_queue.raw_downloader import RawDownloader


class AWSRawDownloader(RawDownloader):
    out_path = None

    def __init__(self, user_id, workspace_id, created_by, **args):
        super().__init__(user_id, workspace_id, created_by, **args)

    def prepare(self):
        pass

    def run_task(self):
        pass
