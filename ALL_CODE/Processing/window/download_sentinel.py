# get products from titles
from collections import OrderedDict 
import os
# Credential
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt

api = SentinelAPI('lehai.ha', 'DangKhoa@123')

titles=[
       
    'S1A_IW_GRDH_1SDV_20190428T222529_20190428T222554_026996_030A09_B6B9',
#     'S1A_IW_GRDH_1SDV_20190510T222529_20190510T222554_027171_03101E_6B8F',
#     'S1A_IW_GRDH_1SDV_20190522T222530_20190522T222555_027346_03159A_11ED',
#     'S1A_IW_GRDH_1SDV_20190603T222531_20190603T222556_027521_031B07_6FAB',
#     'S1A_IW_GRDH_1SDV_20190615T222531_20190615T222556_027696_032050_4E7E',
#     'S1A_IW_GRDH_1SDV_20190627T222532_20190627T222557_027871_032587_71B4',
#     'S1A_IW_GRDH_1SDV_20190709T222533_20190709T222558_028046_032ADD_E0D9'

]
# titles


dir_out = 'Linh_down_descending_subang'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
products=OrderedDict()
for p in titles:
    input_ = dir_out + "/" + p
    print(input_)
    product=api.query_raw('filename:{}.SAFE'.format(input_))
    products.update(product)