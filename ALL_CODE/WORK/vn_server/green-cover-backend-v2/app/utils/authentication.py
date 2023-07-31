import requests

from functools import wraps
from flask import request, current_app

from app.utils.response import unauthorized


def get_user_info():
    URL = current_app.config.get('GEOAI_GATEWAY_AUTH')
    headers = {"Authorization": request.headers["Authorization"]}
    r = requests.get(url=URL, headers=headers)
    headers_json = r.json()
    return headers_json


def get_user_id():
    headers = get_user_info()
    if headers["body"] and headers["body"]["id"]:
        user_id = headers["body"]["id"]
        return user_id
    return None


def auth(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if not request.headers.get("Authorization"):
            unauthorized()
        try:
            user_id = get_user_id()
        except Exception:
            unauthorized() 
        if user_id:
            return f(*args, user_id=user_id, **kwargs)
        unauthorized()
    return wrap
