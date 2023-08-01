from flask import abort
from flask import jsonify


def success(data):
    result = {
        "message": "Sucessful",
        "data": data
    }
    resp = jsonify(result)
    resp.status_code = 200
    return resp


def bad_request(message=None):
    abort(400, description=(message or "Bad request. The parameters weren't satisfied."))


def not_found(errors=[], message='Object not found'):
    abort(404, description=message)


def unauthorized(message='Unauthorized'):
    abort(401, description=message)


def conflict(message='Conflict', type=None):
    if type == "json":
        result = {
            "message": message
        }
        resp = jsonify(result)
        resp.status_code = 409
        abort(resp)
    else:
        abort(409, description=message)
