import shutil

from functools import wraps

from app.job_queue.utils import change_task_status, update_task_process
from app.utils.path import path_2_abs


def task(func):
    @wraps(func)
    def wrap(task_id, *args, **kwargs):
        change_task_status(task_id, 10)
        result = None
        try:
            result = func(task_id, *args, **kwargs)
            change_task_status(task_id, 100)
        except Exception as err:
            change_task_status(task_id, -100, str(err))
        return result
    return wrap


@task
def copy(task_id, srcs, dests):
    count = len(srcs)
    for idx, src_path in enumerate(srcs):
        dest_path = dests[idx]
        abs_src_path = path_2_abs(src_path)
        abs_dest_path = path_2_abs(dest_path)
        shutil.copyfile(abs_src_path, abs_dest_path)
        update_task_process(task_id, (idx+1)/count)
    