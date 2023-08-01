from sqlalchemy import create_engine
from flask_sqlalchemy import SQLAlchemy
from config.db import POSTGIS_DATABASE, POSTGIS_HOST, POSTGIS_PORT, POSTGIS_USER, POSTGIS_PASSWORD
uri = f'postgresql://{POSTGIS_USER}:{POSTGIS_PASSWORD}@{POSTGIS_HOST}:{POSTGIS_PORT}/{POSTGIS_DATABASE}'
engine = create_engine(uri, pool_pre_ping=True)
