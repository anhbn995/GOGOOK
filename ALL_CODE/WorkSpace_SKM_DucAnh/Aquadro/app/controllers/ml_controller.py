from flask import request
import shapely.wkt
from datetime import date
from app.utils.db_service import get_images, get_geometry_from_field
import numpy as np
import joblib
from app.utils.response import success


def _create_zip(label, a1):
    sequence_day = []
    sequence_value = []
    for n in range(len(label)):
        if n == 0:
            sequence_day.append([1, label[n]])
            if a1[0] < a1[-1] and a1[0] >= (a1[-1] / 3) and a1[1] != 0 and a1[-1] != 0:
                sequence_value.append([a1[-1], a1[n]])
            elif a1[0] > a1[-1] and a1[-1] >= (a1[0] / 3) and a1[1] != 0 and a1[-1] != 0:
                sequence_value.append([a1[0], a1[n]])
            else:
                sequence_value.append([0, a1[n]])

        elif n == (len(label) - 1):
            sequence_day.append([label[n - 1], label[n]])
            sequence_day.append([label[n], 366])

            sequence_value.append([a1[n - 1], a1[n]])
            if a1[0] < a1[-1] and a1[0] >= (a1[-1] / 3) and a1[-2] != 0 and a1[0] != 0:
                sequence_value.append([a1[n], a1[-1]])
            elif a1[0] > a1[-1] and a1[-1] >= (a1[0] / 3) and a1[-2] != 0 and a1[0] != 0:
                sequence_value.append([a1[n], a1[0]])
            else:
                sequence_value.append([a1[n], 0])
        else:
            sequence_day.append([label[n - 1], label[n]])
            sequence_value.append([a1[n - 1], a1[n]])
    return sequence_day, sequence_value


def detect_crop():
    payload = request.json
    field = str(payload.get('field'))

    images = get_images(field)

    geo = shapely.wkt.loads(get_geometry_from_field(field))

    # Implement Code của Quyết
    labels = []
    loc_x, loc_y = geo.centroid.xy
    list_stressed = []
    list_slightly = []
    list_moderate = []
    list_healthy = []
    list_very = []

    list_labels = []
    list_labels.append('id_name')
    list_labels.append('location_x')
    list_labels.append('location_y')

    for i in range(1, 366, 1):
        list_labels.append(str(i))

    for i in images:
        year, month, day = i[0].year, i[0].month, i[0].day
        if int(month) == 1:
            num_day = int(day)
        else:
            d0 = date(int(year), 1, 1)
            d1 = date(int(year), int(month), int(day))
            delta = d1 - d0
            num_day = delta.days
        labels.append(num_day)
        list_stressed.append(float(i[1]['NDVI']['statistics'][0]['area']))
        list_slightly.append(float(i[1]['NDVI']['statistics'][1]['area']))
        list_moderate.append(float(i[1]['NDVI']['statistics'][2]['area']))
        list_healthy.append(float(i[1]['NDVI']['statistics'][3]['area']))
        list_very.append(float(i[1]['NDVI']['statistics'][4]['area']))

    a1 = [x + y for (x, y) in zip(list_very, list_healthy)]
    a2 = [x + y for (x, y) in zip(a1, list_moderate)]

    sequence_day, sequence_value = _create_zip(labels, a2)
    list_day = []
    list_value = []
    list_aaa = []
    list_aaa.append(0)
    list_aaa.append(loc_x[0])
    list_aaa.append(loc_y[0])
    for i, j in zip(sequence_day, sequence_value):
        a, b = np.polyfit(i, j, 1)
        for day in range(i[0], i[1], 1):
            value = np.around(((day * a + b) * 100 / max(a1)), 4)
            if value < 0:
                value = 0
            if value > 100:
                value = 100
            list_day.append(day)
            list_value.append(value)
            list_aaa.append(value)

    dict_class = {
        1: "Paddy",
        2: "Sugarcane",
        3: "Maize",
        4: "Cashew",
        5: "Wheat"
    }

    loaded_clf = joblib.load("app/ml_models/classification_agricultural.joblib")
    return success(dict_class[int(loaded_clf.predict(np.asarray([list_aaa[1:]]))[0])])
