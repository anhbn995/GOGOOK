
rq worker train --url redis://localhost:6379 --worker-class app.job_queue.KillWorker --queue-class app.job_queue.KillQueue --job-class app.job_queue.KillJob

rq worker predict --url redis://localhost:6379 --worker-class app.job_queue.KillWorker --queue-class app.job_queue.KillQueue --job-class app.job_queue.KillJob

rq worker utility --url redis://localhost:6379 --worker-class app.job_queue.KillWorker --queue-class app.job_queue.KillQueue --job-class app.job_queue.KillJob

rq worker utility_gpu --url redis://localhost:6379 --worker-class app.job_queue.KillWorker --queue-class app.job_queue.KillQueue --job-class app.job_queue.KillJob

rq worker utility_trick --url redis://localhost:6379 --worker-class app.job_queue.KillWorker --queue-class app.job_queue.KillQueue --job-class app.job_queue.KillJob

rq worker model_predict --url redis://localhost:6379 --worker-class app.job_queue.KillWorker --queue-class app.job_queue.KillQueue --job-class app.job_queue.KillJob