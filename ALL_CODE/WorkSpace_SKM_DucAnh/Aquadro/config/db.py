from config import env

POSTGIS_HOST = env.str('POSTGIS_HOST', '192.168.4.2')
POSTGIS_PORT = env.int('POSTGIS_PORT', 4444)
POSTGIS_DATABASE = env.str('POSTGIS_DATABASE', 'geo-portal')
POSTGIS_USER = env.str('POSTGIS_USER', 'postgres')
POSTGIS_PASSWORD = env('POSTGIS_PASSWORD', 'secret')