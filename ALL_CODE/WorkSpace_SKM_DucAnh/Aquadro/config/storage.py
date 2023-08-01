from config import env
ROOT_DATA_FOLDER = env.str('ROOT_DATA_FOLDER', '/home/geoai/geoai_data_test2/ssr_data')
EOF_ROOT_DATA_FOLDER = env.str('EOF_ROOT_DATA_FOLDER', '/home/geoai/geoai_data_test2/ssr_data')

EODATA_FOLDER = env.str("EODATA_FOLDER", "")
TEMP_FOLDER = env.str('TEMP_FOLDER', '/home/geoai/geoai_data_test2/temp')
DATA_EXPLORER_URL="http://localhost:8002"