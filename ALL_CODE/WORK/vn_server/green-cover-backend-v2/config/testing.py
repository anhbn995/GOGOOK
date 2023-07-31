# -*- coding: utf-8 -*-

from tempfile import mkdtemp

TESTING = True

SQLALCHEMY_DATABASE_URI = "postgres://postgres:admin_123@biztrekk.com:5432/geoai_test"
SQLALCHEMY_ECHO = False
GEOAI_FILES_PATH = mkdtemp(suffix="geoai-uploads")
