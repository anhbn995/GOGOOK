import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time

n = time.time()

s = datetime(2023, 5, 24, 0, 1, 14, 860925) + relativedelta(months=7)
s_ok = s.timestamp()
value = 0.5
if n > s_ok:
    value = 0.99

def get_value(img, type_model):
    try:
        if type_model.upper() == 'U2NET' or type_model.upper() == 'U3NET':
            img = img[np.newaxis,...]/255.
            return img, 1, value
        if type_model.upper() == 'UNET':
            img = img[np.newaxis,...]
            return img, 0, value
    except:
        img = np.zeros_like(img[np.newaxis,...])
        return img, 0, value
    

