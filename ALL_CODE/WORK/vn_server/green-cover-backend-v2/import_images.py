import os.path
import shutil
import ast
import numpy as np
import rasterio
from app.utils.path import make_temp_folder
from inject_image import _translate
from config.default import ROOT_DATA_FOLDER
import uuid
from app.utils import get_file_size
import requests
from config.default import HOSTED_ENDPOINT
from app.utils import statistic
import geopandas as gpd
import json
from sqlalchemy import create_engine
from config.default import SQLALCHEMY_DATABASE_URI
from clip_image import clip
from shapely.geometry import Polygon, MultiPolygon
import gdal, osr, ogr
from vectorize import vectorize_dataset
from import_vectors import store_vectors
from reclass_image import calc_forest


def remove_invalid_gdf(gdf):
    gdf['geometry'] = gdf.buffer(0)

    for ind, g in enumerate(gdf['geometry']):
        if not type(g) in [Polygon, MultiPolygon] or not g.is_valid or g.is_empty:
            if not type(g) in [Polygon, MultiPolygon] or not g.is_valid or g.is_empty:
                gdf = gdf.drop(ind)
    return gdf


def on_success(images):
    payload = []
    for image in images:
        meta = statistic.calculate_metadata(image['path'])
        if "properties" in meta and type(meta['properties']) is str:
            meta['properties'] = ast.literal_eval(meta['properties'])
        tile_url = '/v2/tile/{}/{}/{}'.format(image['year'], image['month'], image['file_id'].hex) + '/{z}/{x}/{y}'
        if "properties" in image:
            meta['properties'] = image['properties']
        if image['type'] == 'classification':
            x_size, y_size = get_pixel_size(image['path'])
            print(49, x_size, y_size)
            with rasterio.open(image['path']) as ds:
                print(np.count_nonzero(ds.read(1) == 1))
                meta['GREEN_COVER_AREA'] = np.count_nonzero(ds.read(1) == 1) * x_size * y_size / 1000000
                meta['WATER_AREA'] = np.count_nonzero(ds.read(1) == 2) * x_size * y_size / 1000000
                meta['NON_GREEN_AREA'] = np.count_nonzero(ds.read(1) == 3) * x_size * y_size / 1000000

        if image['type'] == 'forest_cover_area':
            meta.update(image['meta'])
            # x_size, y_size = get_pixel_size(image['path'])
            # with rasterio.open(image['path']) as ds_forest:
            #     meta['FOREST_COVER_AREA'] = np.count_nonzero(ds_forest.read(1)) * x_size * y_size / 1000000
            #
            #     # meta['NON_FOREST_GREEN_AREA'] = meta['GREEN_COVER_AREA'] - meta['FOREST_COVER_AREA']

        obj = {
            'id': str(image['file_id']),
            'month': image['month'],
            'year': image['year'],
            'name': image['name'],
            'path': f"/{os.path.relpath(image['path'], ROOT_DATA_FOLDER)}",
            'tile_url': tile_url,
            'metadata': meta,
            'size': get_file_size(image['path']),
            'nodata': _get_image_nodata(image['path']),
            'src': image['src'] if "src" in image else None,
            'type': image['type'] if "type" in image else None,
            'aoi_id': image['aoi_id']
        }
        payload.append(obj)

    store_results(payload)


def _get_image_nodata(path):
    with rasterio.open(path) as ds:
        return ds.nodata or 0


def store_results(payload):
    url = '{}/internal/imageries'.format(
        HOSTED_ENDPOINT)
    print(url)
    r = requests.post(url, json=payload)
    if not r.ok:
        print(r.json())

        raise Exception('Fail to upload result')


def statistic_images(image_path, x_size, y_size, band_red=3):
    range_i = [
        [0., 0.2, 0.4, 0.6],
        [0., 0.2, 0.5, 0.8, 1.1],
        [0., 0.2, 0.4, 0.6, 0.8]
    ]
    name_agri_idx = ['ndvi', 'savi']

    name_attr = [
        ['Stressed', 'Slightly Stressed', 'Healthy', 'Very Healthy'],
        ['Open', 'Low Dense', 'Slightly Dense', 'Dense', 'Highly Dense']
    ]
    name_attr_color = [
        ['#FF9800', '#FFEB3B', '#8BC34A', '#4CAF50'],
        ['#F44336', '#3FF980', '#FFEB3B', '#8BC34A', '#4CAF50', ]
    ]

    with rasterio.open(image_path) as dst:

        red = dst.read(band_red).astype(np.float32)
        nir = dst.read(4).astype(np.float32)
        ndvi = (nir - red) / (nir + red).astype(np.float32)
        savi = ((1.5 * (nir - red)) / (red + nir + 0.5)).astype(np.float32)
        out = np.array([ndvi, savi], dtype=np.float32)
        del ndvi
        del savi
        del red
    list_classify = []
    for i in range(len(name_agri_idx)):
        range_class = range_i[i]
        raster_class = out[i]
        image_classify_0 = classify_image(raster_class, range_class)
        list_classify.append(image_classify_0)
        del raster_class
    del out
    agri_properties = {}
    for i in range(len(name_agri_idx)):
        agri_idx = name_agri_idx[i]
        name_attr_index = name_attr[i]
        name_attr_index_color = name_attr_color[i]
        uniqe_index = list(range(2, len(name_attr_index) + 2))
        index_propties = {}
        image_classify = list_classify[i].copy()
        for j in range(len(name_attr_index)):
            index_propties[name_attr_index[j]] = {
                'area': np.count_nonzero(image_classify == uniqe_index[j]) * x_size * y_size / 1000000,
                'color': name_attr_index_color[j]
            }

        agri_properties[agri_idx] = index_propties
    return agri_properties


def classify_image(image_class, range_class):
    classsify = np.zeros(image_class.shape).astype(np.uint8)
    nclass = len(range_class) + 1
    classsify[image_class <= range_class[0]] = 1
    classsify[image_class > range_class[-1]] = nclass
    for i in range(len(range_class) - 1):
        classsify[np.logical_and(image_class > range_class[i], image_class <= range_class[i + 1])] = i + 2
    classsify[image_class == np.float32(0)] = 1
    return classsify.astype(np.uint8)


def reproject_image(src_path, dst_path, dst_crs='EPSG:4326'):
    from osgeo import gdal
    import rasterio
    with rasterio.open(src_path) as ds:
        nodata = ds.nodata or 0
    temp_path = dst_path.replace('.tif', 'temp.tif')
    option = gdal.TranslateOptions(gdal.ParseCommandLine("-co \"TFW=YES\""))
    gdal.Translate(temp_path, src_path, options=option)
    option = gdal.WarpOptions(gdal.ParseCommandLine("-t_srs {} -dstnodata {}".format(dst_crs, nodata)))
    gdal.Warp(dst_path, temp_path, options=option)
    os.remove(temp_path)
    return True


def store_statistic(image_id, payload):
    url = f'{HOSTED_ENDPOINT}/internal/imageries/{image_id}/statistics'
    print(url)
    r = requests.post(url, json=payload)
    if not r.ok:
        raise Exception('Fail to upload result', r.json())


def calc_ndvi(nir_arr, red_arr):
    return (nir_arr - red_arr) / (nir_arr + red_arr)


def reclass_image(mosaic_image, classification_image, out_path, temp_folder):
    dst1 = rasterio.open(classification_image)
    dst2 = rasterio.open(mosaic_image)
    out_profile = dst2.meta
    bandCount = dst2.count
    pre_translate = f'{temp_folder}/reclass_pretranslate.tif'
    dst3 = rasterio.open(pre_translate, 'w', **out_profile)
    for block_index, window in dst2.block_windows(1):
        mask1 = dst1.read(1, window=window)
        accumulator = []
        for i in range(bandCount):
            accumulator += [[mask1]]
        mask = np.vstack(accumulator)
        band = dst2.read(window=window)
        band[mask != 1] = dst2.nodata or 0
        dst3.write(band, window=window)
    dst1.close()
    dst2.close()
    dst3.close()
    _translate(pre_translate, out_path)


def store_mosaic_image(fid_mosaic_view, mosaic_statistic_image, mosaic_view_image, classification_image, year, month,
                       src, initial_aoi, temp_folder):
    aoi_id = initial_aoi['id']
    list_images = []
    dir_path = f'{ROOT_DATA_FOLDER}/data/{year}/{month}/images'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    type_images = [
        {
            'type': 'mosaic',
            'fid': fid_mosaic_view,
            'path': mosaic_view_image,
            'name': 'monthly_image_mosaic',
        },
        {
            'type': 'classification',
            'fid': uuid.uuid4(),
            'properties': {
                "mode": "unique", "bands": [0, 0, 0], "contrastEnhancement": "no",
                "listValue": [{"color": "#228b22", "minValue": 1, "name": "Green cover"},
                              {"color": "#11b9db", "minValue": 2, "name": "Water"},
                              {"color": "#e47a10", "minValue": 3, "name": "Non-green"}],
            },
            'path': classification_image,
            'name': 'green_cover',
        },
        {
            'type': 'mosaic_calc',
            'fid': uuid.uuid4(),
            'path': mosaic_statistic_image,
            'name': 'monthly_image_statistic_mosaic',
        }
    ]
    for type_image in type_images:
        abs_path = f'{dir_path}/{type_image["fid"].hex}.tif'
        reproject_path = f'{temp_folder}/reproject_{type_image["type"]}.tif'
        pre_translate_path = f'{temp_folder}/pre_translate_{type_image["type"]}.tif'
        reproject_image(type_image['path'], reproject_path)
        _translate(reproject_path, pre_translate_path)
        shutil.copy(pre_translate_path, abs_path)
        list_images.append({
            'file_id': type_image['fid'],
            'name': f'{initial_aoi["name"]}_{year}_{month}_{type_image["name"]}',
            'path': abs_path,
            'month': month,
            'year': year,
            'src': src,
            'type': type_image['type'],
            'aoi_id': aoi_id,
            'properties': type_image['properties'] if 'properties' in type_image else None
        })

    return list_images


def calc_vegetable_index(image_path, src, temp_folder, initial_aoi, month, year, classification_image, band_red=3):
    aoi_id = initial_aoi["id"]
    list_images = []
    fid_plant_health = uuid.uuid4()

    fid_density = uuid.uuid4()
    fid_forest = uuid.uuid4()
    pretranslate_out_path = f"{temp_folder}/pretranslate_plant_health.tif"
    pretranslate_density_path = f"{temp_folder}/pretranslate_density.tif"

    ndvi_image = f'{ROOT_DATA_FOLDER}/data/{year}/{month}/images/{fid_plant_health.hex}.tif'
    forest_image = f'{ROOT_DATA_FOLDER}/data/{year}/{month}/images/{fid_forest.hex}.tif'

    with rasterio.open(image_path) as dst:
        red_arr = dst.read(band_red).astype(np.float32)
        nir_arr = dst.read(4).astype(np.float32)
        ndvi = (nir_arr - red_arr) / (nir_arr + red_arr)
        density = ((1.5 * (nir_arr - red_arr)) / (red_arr + nir_arr + 0.5)).astype(np.float32)
        kwargs = dst.meta
    kwargs.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw')

    with rasterio.open(pretranslate_out_path, 'w', **kwargs) as dst:
        dst.write(ndvi.astype(np.float32), 1)
    _translate(pretranslate_out_path, ndvi_image)

    density_image = f'{ROOT_DATA_FOLDER}/data/{year}/{month}/images/{fid_density.hex}.tif'
    with rasterio.open(pretranslate_density_path, 'w', **kwargs) as dst:
        dst.write(density.astype(np.float32), 1)
    _translate(pretranslate_density_path, density_image)

    x_size, y_size = get_pixel_size(classification_image)

    meta_forest = calc_forest(density_image, forest_image, x_size, y_size, classification_image)
    del red_arr
    del nir_arr
    del ndvi
    del density
    list_images.append({
        'file_id': fid_plant_health,
        'path': ndvi_image,
        'name': f'{year}_{month}_plant_health',
        'month': month,
        'year': year,
        'properties': {
            "mode": "pseudocolor", "bands": [0, 0, 0], "contrastEnhancement": "stretch_to_minmax",
            "listValue": [{"color": "#FF9800", "minValue": "0", "name": "Stressed"},
                          {"color": "#FFEB3B", "minValue": "0.2", "name": "Slightly Stressed"},
                          {"color": "#8BC34A", "minValue": "0.4", "name": "Healthy"},
                          {"color": "#4CAF50", "minValue": "0.6", "name": "Very Healthy"}],
        },
        'type': 'plant_health',
        'src': src,
        'aoi_id': aoi_id
    })

    list_images.append({
        'file_id': fid_density,
        'path': density_image,
        'name': f'{initial_aoi["name"]}_{year}_{month}_plant_density',
        'month': month,
        'year': year,
        'properties': {
            "mode": "pseudocolor", "bands": [0, 0, 0], "contrastEnhancement": "stretch_to_minmax",
            "listValue": [{"color": "#F44336", "minValue": "0", "name": "Open"},
                          {"color": "#FF9800", "minValue": "0.2", "name": "Low Dense"},
                          {"color": "#FFEB3B", "minValue": "0.4", "name": "Slightly Dense"},
                          {"color": "#8BC34A", "minValue": "0.6", "name": "Dense"},
                          {"color": "#4CAF50", "minValue": "0.8", "name": "Highly Dense"}
                          ],
        },
        'type': 'green_density',
        'src': src,
        'aoi_id': aoi_id
    })
    list_images.append({
        'file_id': fid_forest,
        'path': forest_image,
        'name': f'{initial_aoi["name"]}_{initial_aoi["name"]}_{year}_{month}_forest_cover',
        'month': month,
        'year': year,
        'properties': {
            "mode": "pseudocolor", "bands": [0, 0, 0], "contrastEnhancement": "stretch_to_minmax",
            "listValue": [
                {"color": "#4CAF50", "minValue": "0.6", "name": "Forest"},
            ],

        },
        "meta": meta_forest,
        'type': 'forest_cover_area',
        'src': src,
        'aoi_id': aoi_id
    })
    return list_images


def calc_area(fid_mosaic, image_path, x_size, y_size, band_red=3):
    properties = statistic_images(image_path, x_size, y_size, band_red)
    print(properties)
    payload = {
        'plant_health': properties['ndvi'],
        'green_density': properties['savi']
    }
    store_statistic(str(fid_mosaic), payload)


def get_pixel_size(image_path):
    temp_folder = make_temp_folder()
    temp_reproject = f'{temp_folder}/3857_mosaic.tif'
    reproject_image(image_path, temp_reproject, 'EPSG:3857')
    with rasterio.open(temp_reproject) as dt:
        x_size = dt.res[0]
        y_size = dt.res[1]
    shutil.rmtree(temp_folder)
    return x_size, y_size


def check_intersection(image_path, aoi_path):
    from shapely.geometry import Polygon
    from shapely.geometry import shape, mapping
    import fiona

    data = gdal.Open(image_path, gdal.GA_ReadOnly)
    geoTransform = data.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * data.RasterXSize
    miny = maxy + geoTransform[5] * data.RasterYSize

    src_projection = osr.SpatialReference(wkt=data.GetProjection())
    tar_projection = osr.SpatialReference()
    tar_projection.ImportFromEPSG(4326)
    wgs84_trasformation = osr.CoordinateTransformation(src_projection, tar_projection)

    point_list = [[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny], [minx, miny]]
    tar_point_list = []

    for _point in point_list:
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(_point[0], _point[1])
        point.Transform(wgs84_trasformation)
        tar_point_list.append([point.GetX(), point.GetY()])

    origin_extend = Polygon(tar_point_list)

    with fiona.open(aoi_path) as ds:
        bounds = []
        for feature in ds:
            aoi_polygon = shape(feature['geometry'])

            bounds.append(aoi_polygon.intersection(origin_extend))

    features = []
    for bound in bounds:
        if not bound.is_empty:
            features.append({
                'type': 'Feature',
                'geometry': mapping(bound)
            })
    if len(features) == 0:
        raise Exception('Not intersect')
    del data
    return features


if __name__ == '__main__':
    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    with engine.connect() as connection:
        aois = connection.execute(f"select id, name, geometry from aois")# where id=14 ")
    list_aoi = []

    for aoi in aois:
        id, name, geometry = aoi
        list_aoi.append({
            'id': id,
            'name': name,
            'geometry': geometry
        })

        print('start import')
        band_red = 3
        for aoi in list_aoi:
            aoi_id = aoi['id']
        initial_images = [
            {
                'src': 'Sentinel',
                'month': '01',
                'year': 2021,

                'mosaic_image': '/home/thang/Desktop/data_npark/T1.tif',
                'classification_image': '/home/thang/Desktop/projects/npark/data/T1.tif',
                'fid_mosaic': uuid.uuid4()
            },
            {
                'src': 'Sentinel',
                'month': '02',
                'year': 2021,

                'mosaic_image': '/home/thang/Desktop/data_npark/T2.tif',
                'classification_image': '/home/thang/Desktop/projects/npark/data/T2.tif',
                'fid_mosaic': uuid.uuid4()
            },
            {
                'src': 'Sentinel',
                'month': '03',
                'year': 2021,

                'mosaic_image': '/home/thang/Desktop/data_npark/T3.tif',
                'classification_image': '/home/thang/Desktop/projects/npark/data/T3.tif',
                'fid_mosaic': uuid.uuid4()
            },
            {
                'src': 'Sentinel',
                'month': '04',
                'year': 2021,

                'mosaic_image': '/home/thang/Desktop/data_npark/T4.tif',
                'classification_image': '/home/thang/Desktop/projects/npark/data/T4.tif',
                'fid_mosaic': uuid.uuid4()
            },
            {
                'src': 'Sentinel',
                'month': '05',
                'year': 2021,
                'mosaic_image': '/home/thang/Desktop/data_npark/T5.tif',
                'classification_image': '/home/thang/Desktop/projects/npark/data/T5.tif',
                'fid_mosaic': uuid.uuid4()
            },
            {
                'src': 'Sentinel',
                'month': '06',
                'year': 2021,
                'mosaic_image': '/home/thang/Desktop/data_npark/T6.tif',
                'classification_image': '/home/thang/Desktop/projects/npark/data/T6.tif',
                'fid_mosaic': uuid.uuid4()
            },
            {
                'src': 'Sentinel',
                'month': '07',
                'year': 2021,
                'mosaic_image': '/home/thang/Desktop/data_npark/T7.tif',
                'classification_image': '/home/thang/Desktop/projects/npark/data/T7.tif',
                'fid_mosaic': uuid.uuid4()
            },
            {
                'src': 'Sentinel',
                'month': '08',
                'year': 2021,
                'mosaic_image': '/home/thang/Desktop/data_npark/T8.tif',
                'classification_image': '/home/thang/Desktop/projects/npark/data/T8.tif',
                'fid_mosaic': uuid.uuid4()
            },
            {
                'src': 'Sentinel',
                'month': '09',
                'year': 2021,
                'mosaic_image': '/home/thang/Desktop/data_npark/T9.tif',
                'classification_image': '/home/thang/Desktop/projects/npark/data/T9.tif',
                'fid_mosaic': uuid.uuid4()
            },
            {
                'src': 'Sentinel',
                'month': '10',
                'year': 2021,
                'mosaic_image': '/home/thang/Desktop/data_npark/T10.tif',
                'classification_image': '/home/thang/Desktop/projects/npark/data/T10.tif',
                'fid_mosaic': uuid.uuid4()
            },
            # {
            #     'src': 'PlanetScope',
            #     'month': '01',
            #     'year': 2021,
            #     'mosaic_image': '/home/thang/Desktop/projects/npark/data/planet/T1.tif',
            #     'classification_image': '/home/thang/Desktop/projects/npark/data/planet/Result_T1.tif',
            #     'fid_mosaic': uuid.uuid4()
            # },
            # {
            #     'src': 'PlanetScope',
            #     'month': '02',
            #     'year': 2021,
            #     'mosaic_image': '/home/thang/Desktop/projects/npark/data/planet/T2.tif',
            #     'classification_image': '/home/thang/Desktop/projects/npark/data/planet/Result_T2.tif',
            #     'fid_mosaic': uuid.uuid4()
            # },
            # {
            #     'src': 'PlanetScope',
            #     'month': '03',
            #     'year': 2021,
            #     'mosaic_image': '/home/thang/Desktop/projects/npark/data/planet/T3.tif',
            #     'classification_image': '/home/thang/Desktop/projects/npark/data/planet/Result_T3.tif',
            #     'fid_mosaic': uuid.uuid4()
            # },
            # {
            #     'src': 'PlanetScope',
            #     'month': '04',
            #     'year': 2021,
            #     'mosaic_image': '/home/thang/Desktop/projects/npark/data/planet/T4.tif',
            #     'classification_image': '/home/thang/Desktop/projects/npark/data/planet/Result_T4.tif',
            #     'fid_mosaic': uuid.uuid4()
            # },
            # {
            #     'src': 'PlanetScope',
            #     'month': '05',
            #     'year': 2021,
            #     'mosaic_image': '/home/thang/Desktop/projects/npark/data/planet/T5.tif',
            #     'classification_image': '/home/thang/Desktop/projects/npark/data/planet/Result_T5.tif',
            #     'fid_mosaic': uuid.uuid4()
            # },
            # {
            #     'src': 'PlanetScope',
            #     'month': '06',
            #     'year': 2021,
            #     'mosaic_image': '/home/thang/Desktop/projects/npark/data/planet/T6.tif',
            #     'classification_image': '/home/thang/Desktop/projects/npark/data/planet/Result_T6.tif',
            #     'fid_mosaic': uuid.uuid4()
            # },
            # {
            #     'src': 'PlanetScope',
            #     'month': '07',
            #     'year': 2021,
            #     'mosaic_image': '/home/thang/Desktop/projects/npark/data/planet/T7.tif',
            #     'classification_image': '/home/thang/Desktop/projects/npark/data/planet/Result_T7.tif',
            #     'fid_mosaic': uuid.uuid4()
            # },
            # {
            #     'src': 'PlanetScope',
            #     'month': '08',
            #     'year': 2021,
            #     'mosaic_image': '/home/thang/Desktop/projects/npark/data/planet/T8.tif',
            #     'classification_image': '/home/thang/Desktop/projects/npark/data/planet/Result_T8.tif',
            #     'fid_mosaic': uuid.uuid4()
            # },
            # {
            #     'src': 'PlanetScope',
            #     'month': '09',
            #     'year': 2021,
            #     'mosaic_image': '/home/thang/Desktop/projects/npark/data/planet/T9.tif',
            #     'classification_image': '/home/thang/Desktop/projects/npark/data/planet/Result_T9.tif',
            #     'fid_mosaic': uuid.uuid4()
            # },
            # {
            #     'src': 'PlanetScope',
            #     'month': '10',
            #     'year': 2021,
            #     'mosaic_image': '/home/thang/Desktop/projects/npark/data/planet/T10.tif',
            #     'classification_image': '/home/thang/Desktop/projects/npark/data/planet/Result_T10.tif',
            #     'fid_mosaic': uuid.uuid4()
            # },
            # {
            #     'src': 'Jilin',
            #     'month': '01',
            #     'year': 2021,
            #     'mosaic_image': '/home/thang/Downloads/Jan 2021 JL1GF03B04_PMS_20210122101129_200038368_102_0001_001_L1.tif',
            #     'classification_image': '/home/thang/Downloads/Jan 2021 JL1GF03B04_PMS_20210122101129_200038368_102_0001_001_L1 (1).tif',
            #     'fid_mosaic': uuid.uuid4()
            # },
            # {
            #     'src': 'Jilin',
            #     'month': '02',
            #     'year': 2021,
            #     'mosaic_image': '/home/thang/Downloads/Feb 2021 JL1GF03B04_PMS_20210225102015_200042310_103_0002_001_L1.tif',
            #     'classification_image': '/home/thang/Downloads/Feb 2021 JL1GF03B04_PMS_20210225102015_200042310_103_0002_001_L1 (1).tif',
            #     'fid_mosaic': uuid.uuid4()
            # }

        ]

        gdf = gpd.GeoDataFrame.from_features(aoi["geometry"])
        gdf = remove_invalid_gdf(gdf)

        try:
            for initial_image in initial_images:
                list_images = []
                temp_folder = make_temp_folder()
                month = initial_image['month']
                year = initial_image['year']
                init_mosaic_image = initial_image['mosaic_image']
                init_image = initial_image['classification_image']

                mosaic_image = f'{temp_folder}/mosaic_crop.tif'

                new_aois_path = f"{temp_folder}/temp_aoi.json"
                with open(new_aois_path, 'w') as json_file:
                    json.dump(json.loads(gdf.to_json()), json_file)
                try:
                    new_aois = check_intersection(init_image, new_aois_path)
                except Exception as e:
                    print(e)
                    continue
                aoi['geometry'] = {
                    "type": "FeatureCollection",
                    "features": new_aois
                }
                clip(init_mosaic_image, aoi['geometry'], mosaic_image, temp_folder)

                classification_image = f'{temp_folder}/classification_crop.tif'

                clip(init_image, aoi['geometry'], classification_image, temp_folder)

                fid_mosaic = initial_image['fid_mosaic']
                src = initial_image['src']
                pre_image = None

                if len(list_images) > 0:
                    pre_image = list_images[len(list_images) - 1]
                list_image = store_mosaic_image(fid_mosaic, mosaic_image, classification_image, year,
                                                month, src,
                                                initial_aoi, temp_folder)
                list_images += list_image
                reclass_image_path = f'{temp_folder}/reclass_mosaic.tif'

                reclass_image(mosaic_image, classification_image, reclass_image_path, temp_folder)

                list_images += calc_vegetable_index(reclass_image_path, src, temp_folder, aoi_id, month, year,
                                                    reclass_image_path)
                # import pdb
                # pdb.set_trace()
                on_success(list_images)

                print("Done store mosaic, classification, vegetable index")
                x_size, y_size = get_pixel_size(mosaic_image)
                print(x_size, y_size)

                calc_area(fid_mosaic, reclass_image_path, x_size, y_size)
                print("Done cacl area")

        except Exception as e:
            raise e
        finally:
            shutil.rmtree(temp_folder)
        print("Done aoi_id ", aoi['id'])
