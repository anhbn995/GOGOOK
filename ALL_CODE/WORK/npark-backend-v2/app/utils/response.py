from flask import abort
from flask import jsonify


def success(data):
    result = {
        "message": "Successful",
        "data": data
    }
    resp = jsonify(result)
    print(result)
    resp.status_code = 200
    return resp



def need_to_by_processing_px(message=None):
    abort(410, description=message or 'Do not have enough EOToken to run predict')


def pay_more(message=None):
    abort(409, description=message or 'Please upgrade your plan!')


def model_image_not_suitable():
    abort(500, description="This image cannot be predicted with given model.")


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


def error(message='Error'):
    result = {
        "message": message
    }
    resp = jsonify(result)
    resp.status_code = 500
    return resp

