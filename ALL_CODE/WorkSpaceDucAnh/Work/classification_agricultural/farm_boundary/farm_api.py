# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 11:24:51 2022

@author: AnhHo
"""
from flask_restful import reqparse, abort, Api, Resource
from flask import request, abort, Flask
from threading import Thread
app = Flask(__name__)
api = Api(app)
app_port = 6790


class Upload_tiles(Resource):
    def post(self):
        file = request.files['file']
        min_lat = request.form['min_lat']
        min_long = request.form['min_long']
        max_lat = request.form['max_lat']
        max_long = request.form['max_long']
        zoom_level = request.form['zoom_level']
        print(min_lat)
        return True


def cal_farm_process():
    pass


if __name__ == '__main__':
    t1 = Thread(target=cal_farm_process, args=())
    t1.daemon = True
    t1.start()

    api.add_resource(Upload_tiles, '/submit')
    app.run(debug=0,host="0.0.0.0",port=app_port)
