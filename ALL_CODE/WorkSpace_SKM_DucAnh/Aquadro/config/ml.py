GREEN_WEIGHTS="/home/geoai/geoai_data_test2/tmp_ducanh/adaro/weight/green_weights.h5"
FOREST_WEIGHTS_V2="/home/geoai/geoai_data_test2/tmp_ducanh/adaro/weight/forest_weights_v2.h5"
WATER_WEIGHTS="/home/geoai/geoai_data_test2/tmp_ducanh/adaro/weight/water_weights.h5"
CLOUD_ONLY="/home/geoai/geoai_data_test2/tmp_ducanh/adaro/weight/cloud/cloud_only.h5"
SHADOW_ONLY="/home/geoai/geoai_data_test2/tmp_ducanh/adaro/weight/cloud/shadow_only.h5"

DICT_COLORMAP =  {
                    0: (0,0,0, 0), # Nodata
                    1: (0,255,0,0), # Green
                    2: (100, 149, 237, 0), # water
                    3: (101,67,33, 0), # BuildUp
                    4: (0,128,0, 0), # Forest
                    5: (255,255,255,0) # Cloud
                }

WIEGHT_STORAGE = {
        'cloud': CLOUD_ONLY,
        'shadow': SHADOW_ONLY,
        'green': GREEN_WEIGHTS,
        'forest': FOREST_WEIGHTS_V2,
        'water': WATER_WEIGHTS
    }