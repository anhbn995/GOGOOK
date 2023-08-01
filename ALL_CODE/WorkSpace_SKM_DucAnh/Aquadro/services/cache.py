from redis import Redis
from config.cache import REDIS_CACHE_PORT, REDIS_CACHE_HOST
redis = Redis(host=REDIS_CACHE_HOST, port=REDIS_CACHE_PORT)
