from flask_caching import Cache

cache = Cache(config={'CACHE_TYPE': 'redis_queue', 'CACHE_REDIS_URL': 'redis_queue://localhost:6379'})
