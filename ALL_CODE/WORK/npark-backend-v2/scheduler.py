import functools
from time import sleep
import schedule
import threading


def bad_task():
    sleep(5)
    print("done")
    raise Exception("Error!")


class ScheduleThread(threading.Thread):

    def run(self):
        self.exc = None
        try:
            if self._target:
                self._target(*self._args, **self._kwargs)
        except Exception as e:
            self.exc = e
        finally:
            del self._target, self._args, self._kwargs

    def join(self):
        super().join()
        if self.exc:
            raise self.exc


def job(callable, retries=0, run_once=False, retry_after=5, retry_job=False):
    t = ScheduleThread(target=callable, daemon=True)
    t.start()
    print(
        f"Running {'retry job' if retry_job else 'job'} in thread {t.ident}...:")
    try:
        t.join()
        print("Job run successfully")
    except Exception as e:
        print(e)
        if retries:
            print(
                f"Retry job after {retry_after} seconds, remaining: {retries}")
            schedule.every(retry_after).seconds.do(functools.partial(
                job, callable, retries=retries-1, run_once=True, retry_job=True))
    finally:
        if run_once:
            return schedule.CancelJob


def task():
    from run_download_to_predict import main
    from datetime import date
    if date.today().day == 1:
        main()


schedule.every().day.at("00:00").do(
    functools.partial(job, task, retries=3, retry_after=120))
print("Scheduler is started....")
while True:
    schedule.run_pending()
    sleep(1)
