from config import env
worker_hijack_root_logger = False
worker_redirect_stdouts_level = 'INFO'
worker_concurrency = env.int("WORKER_CONCURRENCY", 1)
worker_hijack_root_logger = False
broker_transport_options = {"visibility_timeout": 3600*24}
