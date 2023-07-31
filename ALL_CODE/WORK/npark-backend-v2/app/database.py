from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from flask import current_app


def query(cls, **kw):
    q = db.session.query(cls)

    if kw:
        q = q.filter_by(**kw)

    return q


def get(cls, id):
    return cls.query().get(id)


def exists(cls, **kw):
    return cls.query(**kw).first() is not None


def apply_kwargs(self, kwargs):
    for key, value in kwargs.items():
        setattr(self, key, value)

    return self


def create_db_session(uri=None):
    if not uri:
        database_connection_info = current_app.config.get('SQLALCHEMY_DATABASE_URI')
    else: database_connection_info = uri
    db_engine = create_engine(database_connection_info, echo=False)
    scoped_db = scoped_session(
        sessionmaker(
            autoflush=True,
            autocommit=False,
            bind=db_engine
        )
    )
    return scoped_db


db = SQLAlchemy(session_options=dict(expire_on_commit=False))

db.Model.flask_query = db.Model.query
db.Model.query = classmethod(query)
db.Model.get = classmethod(get)
db.Model.exists = classmethod(exists)
db.Model.apply_kwargs = apply_kwargs
