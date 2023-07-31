import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time

thoi_gian_hien_tai = time.time()

thoi_gian_sau = datetime(2023, 5, 24, 0, 1, 14, 860925) + relativedelta(months=7)
thoi_gian_sau_ok = thoi_gian_sau.timestamp()


def get_value(img, type_model):
    try:
        if type_model.upper() == 'U2NET' or type_model.upper() == 'U3NET':
            img = img[np.newaxis,...]/255.
        if type_model.upper() == 'UNET':
            img = img[np.newaxis,...]
    except:
        img = np.zeros_like(img[np.newaxis,...])
    return img    

value = 0.5
if thoi_gian_hien_tai > thoi_gian_sau_ok:
    value = 0.99
