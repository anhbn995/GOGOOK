from os import environ
from sentinelsat import SentinelAPI
from datetime import timedelta
from config.default import SENTINEL_API_USER, SENTINEL_API_PASSWORD, SENTINEL_API_URL

def convert_l1c_to_l2a(sen2lv1_ids):
    api = SentinelAPI(SENTINEL_API_USER, SENTINEL_API_PASSWORD, SENTINEL_API_URL)
    response = []
    for sen2lv1_id in sen2lv1_ids:
        try: 
            arr = sen2lv1_id.split('_')
            # get metadata
            products = api.query(filename=sen2lv1_id + '.SAFE')
            metadata = list(products.items())[0][1]
            # get time period
            d1 = metadata['beginposition']
            d2 = d1 + timedelta(hours=1)
            # get level1 identifier
            level1cpdiidentifier = metadata.get('level1cpdiidentifier')
            if level1cpdiidentifier:
                query = {
                    'date': (d1, d2),
                    'platformname': 'Sentinel-2',
                    'filename': arr[0] + '_MSIL2A_' + arr[2] + '*' + arr[5] + '_*',
                    'level1cpdiidentifier': level1cpdiidentifier
                }
            else:
                query = {
                    'date': (d1, d2),
                    'platformname': 'Sentinel-2',
                    'filename': arr[0] + '_MSIL2A_' + arr[2] + '*' + arr[5] + '_*'
                }
            # query level 2

            products = api.query(**query)
            items = list(products.items())
            if len(items) > 0:
                response.append({
                    'L1C': sen2lv1_id,
                    'L2A': items[0][1]['title']
                })
            else:
                response.append({
                    'L1C': sen2lv1_id,
                    'L2A': None
                })
        except:
            response.append({
                'L1C': sen2lv1_id,
                'L2A': None
            })
    return response