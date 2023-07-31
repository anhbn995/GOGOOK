import uuid

import rasterio
import xarray as xr
import numpy as np
import time
from inject_image import _translate
from import_images import on_success
from config.default import ROOT_DATA_FOLDER
import os


# client = Client("tcp://192.168.4.139:8786")
def _calc_slope_poly(y):
    """ufunc to be used by linear_trend"""
    x = np.arange(len(y))
    return np.polyfit(x, y, 1)[0]


def slop_ndvi(list_path_file, out_path):
    a = time.time()
    array_ndvi = []

    for img in list_path_file:
        with rasterio.open(img['path']) as dst:
            kwargs = dst.meta
        image = xr.open_rasterio(img['path'], chunks={'band': 1, 'x': 6000, 'y': 6000})
        image = image[0]
        image.expand_dims({'time': 1})
        image['time'] = img['time']
        array_ndvi.append(image)

    result_ndvi = xr.concat(array_ndvi, dim='time')
    result_ndvi = result_ndvi.chunk({'time': len(result_ndvi['time']), 'x': 6000, 'y': 6000})

    result_ndvi = result_ndvi.compute()
    # data = result_ndvi.polyfit(dim='time', deg=1, skipna=True, full=True)
    result_ndvi = result_ndvi.fillna(0)
    data = xr.apply_ufunc(_calc_slope_poly, result_ndvi,
                          vectorize=True,
                          input_core_dims=[['time']],
                          output_dtypes=[np.float],
                          dask='parallelized')
    data.compute()

    kwargs.update(dtype=rasterio.float64, count=1, compress='lzw')
    with rasterio.open(out_path, 'w', **kwargs) as dst:
        dst.write(data, 1)
    print(time.time() - a)

    return data


if __name__ == '__main__':
    list_path_file = [
        {
            'path': f'{ROOT_DATA_FOLDER}/data/2021/01/images/d6f7936a53ad46319901fbd93691421b.tif',
            'time': np.datetime64('2021-01')
        },
        {
            'path': f'{ROOT_DATA_FOLDER}/data/2021/02/images/a358460b8a0644708fc9f65ca98a44fb.tif',
            'time': np.datetime64('2021-02')
        },
        {
            'path': f'{ROOT_DATA_FOLDER}/data/2021/03/images/2ec442a516aa463d8e52bbaf9613fc54.tif',
            'time': np.datetime64('2021-03')
        },
        {
            'path': f'{ROOT_DATA_FOLDER}/data/2021/04/images/2ab1b6d426db46faad272fa858b54c76.tif',
            'time': np.datetime64('2021-04')
        },
        {
            'path': f'{ROOT_DATA_FOLDER}/data/2021/05/images/8f0d30efb4314d948b71d9938fd539a3.tif',
            'time': np.datetime64('2021-05')
        },
        {
            'path': f'{ROOT_DATA_FOLDER}/data/2021/06/images/7cb3e67c67ee4068bdb5d2f79e0b1e11.tif',
            'time': np.datetime64('2021-06')
        },
        {
            'path': f'{ROOT_DATA_FOLDER}/data/2021/07/images/417459df8db7487588072361b0c1ad02.tif',
            'time': np.datetime64('2021-07')
        },
        {
            'path': f'{ROOT_DATA_FOLDER}/data/2021/08/images/163f881f405d43d5ae5eaaafad928b4c.tif',
            'time': np.datetime64('2021-08')
        },
        {
            'path': f'{ROOT_DATA_FOLDER}/data/2021/09/images/90543604657f469a992b8275689d252d.tif',
            'time': np.datetime64('2021-09')
        },
    ]
    year = 2021
    fid = uuid.uuid4()
    month = '00'
    pre_translate_path = '/home/thang/Downloads/result.tif'
    dir_path = f'{ROOT_DATA_FOLDER}/data/{year}/{month}/images'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    out_path = f'{dir_path}/{fid.hex}.tif'
    # data = slop_ndvi(list_path_file, pre_translate_path)
    image_classification = f'{ROOT_DATA_FOLDER}/data/2021/09/images/90543604657f469a992b8275689d252d.tif'
    with rasterio.open(image_classification) as dst:
        band = dst.read(1)

    with rasterio.open(pre_translate_path) as dst1:
        band1 = dst1.read(1)
    out_profile = dst1.meta

    band1[band != 1] = dst1.nodata or 0

    temp_path = f'/home/thang/Downloads/reclass_pretranslate.tif'

    dst3 = rasterio.open(temp_path, 'w', **out_profile)
    dst3.write(band1, 1)
    dst3.close()

    _translate(temp_path, out_path)

    list_images = []
    list_images.append({
        'file_id': fid,
        'path': out_path,
        'name': f'{year}_slope_plant_health',
        'month': month,
        'year': year,
        'properties': {
            "mode": "pseudocolor", "bands": [0, 0, 0], "contrastEnhancement": "stretch_to_minmax",
            "listValue": [{"color": "#d32f6f", "minValue": "-0.2", "name": "-0.2 to -0.02"},
                          {"color": "#ff5722", "minValue": "-0.02", "name": "-0.02 to 0.05"},
                          {"color": "#ffeb3b", "minValue": "0.05", "name": "0.05 to 0.2"},
                          {"color": "#4CAF50", "minValue": "0.2", "name": " > 0.2"}]

        },
        'type': 'slope',
        'src': 'Sentinel',
        'aoi_id': None
    })

    on_success(list_images)
