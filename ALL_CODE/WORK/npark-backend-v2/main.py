from app.utils.path import make_temp_folder
import uuid
import geopandas as gpd
import json
from sqlalchemy import create_engine
from config.default import SQLALCHEMY_DATABASE_URI
from clip_image import clip
import shutil
from import_images import remove_invalid_gdf, check_intersection, store_mosaic_image, reclass_image, \
    calc_vegetable_index, on_success, get_pixel_size, calc_area
from vectorize import vectorize_image
from import_vectors import store_cloud
from app.models.imagery import Imagery
import datetime
from dateutil.relativedelta import relativedelta
import os

AOI_SINGAPORE = 14
SOURCE_TYPES = [
    {
        'id': 1,
        'key': 'Sentinel',
        'band_red': 3
    },
    {
        'id': 2,
        'key': 'PlanetScope',
        'band_red': 3
    },
    {
        'id': 3,
        'key': 'Jilin',
        'band_red': 3
    }
]


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


def main(image_mosaic_statistic_path: str, image_mosaic_view_path: str, image_classification_path: str, cloud_path: str,
         src: str, month: str, before_month, year: int, before_year):
    source_image = next((item for item in SOURCE_TYPES if item["key"] == src), None)
    band_red = source_image['band_red']
    temp_folder = make_temp_folder()
    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    with engine.connect() as connection:
        aois = connection.execute(f"select id, name, geometry from aois")  # where id =14
    list_aoi = []
    try:
        for aoi in aois:

            id, name, geometry = aoi
            list_aoi.append({
                'id': id,
                'name': name,
                'geometry': geometry
            })
            initial_aoi = {
                'id': id,
                'name': name,
                'geometry': geometry
            }
            for aoi in list_aoi:
                aoi_id = aoi['id']

            print(111111, aoi['id'])
            # print(initial_aoi)
            gdf = gpd.GeoDataFrame.from_features(aoi["geometry"])

            gdf = remove_invalid_gdf(gdf)

            mosaic_image_statistic_reproject = f'{temp_folder}/mosaic_image_statistic_reproject.tif'
            reproject_image(image_mosaic_statistic_path, mosaic_image_statistic_reproject)
            # image_mosaic_statistic_path = mosaic_image_statistic_reproject

            mosaic_image_view_reproject = f'{temp_folder}/mosaic_image_view_reproject.tif'
            reproject_image(image_mosaic_view_path, mosaic_image_view_reproject)
            # image_mosaic_view_path = mosaic_image_view_reproject

            image_classification_reproject = f'{temp_folder}/mosaic_image_classification_reproject.tif'
            reproject_image(image_classification_path, image_classification_reproject)
            # image_classification_path = image_classification_reproject

            mosaic_image_statistic_reproject = f'{temp_folder}/mosaic_image_statistic_reproject.tif'
            reproject_image(image_mosaic_statistic_path, mosaic_image_statistic_reproject)
            # image_mosaic_statistic_path = mosaic_image_statistic_reproject

            list_images = []
            init_mosaic_image = image_mosaic_statistic_path
            init_image = image_classification_path
            mosaic_image_statistic_crop = f'{temp_folder}/mosaic_crop_statistic.tif'
            mosaic_image_view_crop = f'{temp_folder}/mosaic_crop_view.tif'

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
            clip(init_mosaic_image, aoi['geometry'], mosaic_image_statistic_crop, temp_folder)
            clip(image_mosaic_view_path, aoi['geometry'], mosaic_image_view_crop, temp_folder)

            classification_image = f'{temp_folder}/classification_crop.tif'
            clip(init_image, aoi['geometry'], classification_image, temp_folder)

            fid_mosaic = uuid.uuid4()
            list_image = store_mosaic_image(fid_mosaic, mosaic_image_statistic_crop, mosaic_image_view_crop,
                                            classification_image, year,
                                            month, src,
                                            initial_aoi, temp_folder)
            list_images += list_image
            reclass_image_path = f'{temp_folder}/reclass_mosaic.tif'

            reclass_image(mosaic_image_statistic_crop, classification_image, reclass_image_path, temp_folder)
            list_images += calc_vegetable_index(reclass_image_path, src, temp_folder, initial_aoi, month, year,
                                                classification_image, band_red)
            print(list_images)
            on_success(list_images)

            print("Done store mosaic, classification, vegetable index")
            x_size, y_size = get_pixel_size(mosaic_image_statistic_crop)
            print(x_size, y_size)

            calc_area(fid_mosaic, reclass_image_path, x_size, y_size, band_red)
            print("Done cacl area")

            print("Done aoi_id ", aoi['id'])
            if before_month:
                vectorize_image(temp_folder, source_image, month, before_month, year, before_year, initial_aoi)
            if cloud_path:
                store_cloud(cloud_path, month, year, source_image, initial_aoi)
    except Exception as e:
        raise e
    finally:
        shutil.rmtree(temp_folder)


def update_or_create_image(image_mosaic_statistic_path: str, image_mosaic_view_path: str,
                           image_classification_path: str, cloud_path: str,
                           start_date: str, end_date: str, src='Sentinel'):
    start = datetime.strptime(start_date, '%d-%m-%Y')
    month = start.month
    year = start.year
    main(image_mosaic_statistic_path, image_mosaic_view_path, image_classification_path, cloud_path, src, month,
         before_month, year, before_year)


if __name__ == '__main__':
    # thang_11 = {
    #     "statistic_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T11.tif',
    #     "view_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T11.tif',
    #     "result_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/classification/2021_T11.tif',
    #     "cloud_path": "/home/geoai/geoai_data_test2/data_npark/Sentinel/cloud/2021_T11.shp",
    #     "month": '11',
    #     "year": 2021
    # }
    # thang_12 = {
    #     "statistic_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T12.tif',
    #     "view_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T12.tif',
    #     "result_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/classification/2021_T12.tif',
    #     "cloud_path": "/home/geoai/geoai_data_test2/data_npark/Sentinel/cloud/2021_T12.shp",
    #     "month": '12',
    #     "year": 2021
    # }
    #
    # thang_1_2022 = {
    #     "statistic_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2022_T1.tif',
    #     "view_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2022_T1.tif',
    #     "result_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/classification/2022_T1.tif',
    #     "cloud_path": "/home/geoai/geoai_data_test2/data_npark/Sentinel/cloud/2022_T1.shp",
    #     "month": '01',
    #     "year": 2022
    # }
    # thang_2_2022 = {
    #     "statistic_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2022_T2.tif',
    #     "view_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2022_T2.tif',
    #     "result_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/classification/2022_T2.tif',
    #     "cloud_path": "/home/geoai/geoai_data_test2/data_npark/Sentinel/cloud/2022_T2.shp",
    #     "month": '02',
    #     "year": 2022
    # }

    thang_3_2022 = {
        "statistic_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2022_T3.tif',
        "view_mosaic_path": '/home/skm/SKM_OLD/data_ml_mount/Quyet Workspace/Green Cover/Green Cover Npark Singapore/results/T3-2022/T3-2022_new.tif',
        "result_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/classification/2022_T3.tif',
        "cloud_path": "/home/geoai/geoai_data_test2/data_npark/Sentinel/cloud/2022_T3.shp",
        "month": '03',
        "year": 2022
    }
    main(thang_3_2022['statistic_mosaic_path'], thang_3_2022['view_mosaic_path'], thang_3_2022['result_path'],
         thang_3_2022['cloud_path'], 'Sentinel', '03', '02', 2022, 2022)

    # main(thang_12['statistic_mosaic_path'], thang_12['view_mosaic_path'], thang_12['result_path'],
    #      thang_12['cloud_path'], 'Sentinel', '12', '11', 2021, 2021)

    sentinels = [
        {
            "statistic_mosaic_path": '/home/thang/Downloads/2021_T11 (1).tif',
            "view_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T1.tif',
            "result_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/classification/2021_T11.tif',
            "cloud_path": "/home/geoai/geoai_data_test2/data_npark/Sentinel/cloud/2021_T1.shp",
            "month": '01',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T2.tif',
            "view_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T2.tif',
            "result_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/classification/2021_T2.tif',
            "cloud_path": "/home/geoai/geoai_data_test2/data_npark/Sentinel/cloud/2021_T2.shp",
            "month": '02',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T3.tif',
            "view_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T3.tif',
            "result_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/classification/2021_T3.tif',
            "cloud_path": "/home/geoai/geoai_data_test2/data_npark/Sentinel/cloud/2021_T3.shp",
            "month": '03',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T4.tif',
            "view_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T4.tif',
            "result_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/classification/2021_T4.tif',
            "cloud_path": "/home/geoai/geoai_data_test2/data_npark/Sentinel/cloud/2021_T4.shp",
            "month": '04',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T5.tif',
            "view_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T5.tif',
            "result_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/classification/2021_T5.tif',
            "cloud_path": "/home/geoai/geoai_data_test2/data_npark/Sentinel/cloud/2021_T5.shp",
            "month": '05',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T6.tif',
            "view_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T6.tif',
            "result_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/classification/2021_T6.tif',
            "cloud_path": "/home/geoai/geoai_data_test2/data_npark/Sentinel/cloud/2021_T6.shp",
            "month": '06',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T7.tif',
            "view_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T7.tif',
            "result_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/classification/2021_T7.tif',
            "cloud_path": "/home/geoai/geoai_data_test2/data_npark/Sentinel/cloud/2021_T7.shp",
            "month": '07',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T8.tif',
            "view_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T8.tif',
            "result_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/classification/2021_T8.tif',
            "cloud_path": "/home/geoai/geoai_data_test2/data_npark/Sentinel/cloud/2021_T8.shp",
            "month": '08',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T9.tif',
            "view_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T9.tif',
            "result_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/classification/2021_T9.tif',
            "cloud_path": "/home/geoai/geoai_data_test2/data_npark/Sentinel/cloud/2021_T9.shp",
            "month": '09',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T10.tif',
            "view_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T10.tif',
            "result_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/classification/2021_T10.tif',
            "cloud_path": "/home/geoai/geoai_data_test2/data_npark/Sentinel/cloud/2021_T10.shp",
            "month": '10',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T11.tif',
            "view_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T11.tif',
            "result_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/classification/2021_T11.tif',
            "cloud_path": "/home/geoai/geoai_data_test2/data_npark/Sentinel/cloud/2021_T11.shp",
            "month": '11',
            "year": 2021
        },
    ]
    thang11 = {
        "statistic_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T11.tif',
        "view_mosaic_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/2021_T11.tif',
        "result_path": '/home/geoai/geoai_data_test2/data_npark/Sentinel/classification/2021_T11.tif',
        "cloud_path": "/home/geoai/geoai_data_test2/data_npark/Sentinel/cloud/2021_T11.shp",
        "month": '11',
        "year": 2021
    }

    # main(thang11['statistic_mosaic_path'], thang11['view_mosaic_path'], thang11['result_path'],
    #      thang11['cloud_path'], 'Sentinel', '11', '10', 2021, 2021)

    # for i, image in enumerate(sentinels):
    #     before_month = sentinels[i - 1]['month'] if i > 0 else None
    #     main(image['statistic_mosaic_path'], image['view_mosaic_path'], image['result_path'], image['cloud_path'],
    #          'Sentinel', image['month'], before_month, image['year'], image['year'])

    PlanetScopes = [
        {
            'month': '01',
            'year': 2021,
            'statistic_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/mosaic/2021_T1.tif',
            'view_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/mosaic/2021_T1.tif',
            'result_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/classification/2021_T2.tif',
        },
        {
            'month': '02',
            'year': 2021,
            'statistic_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/mosaic/2021_T2.tif',
            'view_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/mosaic/2021_T2.tif',
            'result_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/classification/2021_T2.tif',
        },
        {
            'month': '03',
            'year': 2021,
            'statistic_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/mosaic/2021_T3.tif',
            'view_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/mosaic/2021_T3.tif',
            'result_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/classification/2021_T3.tif',
        },
        {
            'month': '04',
            'year': 2021,
            'statistic_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/mosaic/2021_T4.tif',
            'view_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/mosaic/2021_T4.tif',
            'result_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/classification/2021_T4.tif',
        },
        {
            'month': '05',
            'year': 2021,
            'statistic_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/mosaic/2021_T5.tif',
            'view_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/mosaic/2021_T5.tif',
            'result_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/classification/2021_T5.tif',
        },
        {
            'month': '06',
            'year': 2021,
            'statistic_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/mosaic/2021_T6.tif',
            'view_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/classification/2021_T6.tif',

        },
        {
            'month': '07',
            'year': 2021,
            'statistic_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/mosaic/2021_T7.tif',
            'view_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/mosaic/2021_T7.tif',
            'result_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/classification/2021_T7.tif',

        },
        {
            'month': '08',
            'year': 2021,
            'statistic_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/mosaic/2021_T8.tif',
            'view_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/mosaic/2021_T8.tif',
            'result_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/classification/2021_T8.tif',

        },
        {
            'month': '09',
            'year': 2021,
            'statistic_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/mosaic/2021_T9.tif',
            'view_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/mosaic/2021_T9.tif',
            'result_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/classification/2021_T9.tif',

        },
        {
            'month': '10',
            'year': 2021,
            'statistic_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/mosaic/2021_T10.tif',
            'view_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/mosaic/2021_T10.tif',
            'result_path': '/home/geoai/geoai_data_test2/data_npark/PlanetScope/classification/2021_T10.tif',
        },
    ]
    # print("Done sentinel 2")
    # for i, image in enumerate(PlanetScopes):
    #     before_month = PlanetScopes[i - 1]['month'] if i > 0 else None
    #     main(image['statistic_mosaic_path'], image['view_mosaic_path'],image['result_path'], None, 'PlanetScope', image['month'], before_month,
    #          image['year'],image['year'])

    jilins = [
        {
            'month': '01',
            'year': 2021,
            'statistic_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/Jilin/mosaic/2021_T1.tif',
            'result_path': '/home/geoai/geoai_data_test2/data_npark/Jilin/classification/2021_T1.tif',
        },
        {
            'month': '02',
            'year': 2021,
            'statistic_mosaic_path': '/home/geoai/geoai_data_test2/data_npark/Jilin/mosaic/2021_T2.tif',
            'result_path': '/home/geoai/geoai_data_test2/data_npark/Jilin/classification/2021_T2.tif',
        }
    ]

    sentinels_kolkata = [
        {
            "statistic_mosaic_path": '/home/thang/Desktop/Kolkata/T1/mosaic_statistic/T1_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/Kolkata/T1/mosaic_statistic/T1_DA.tif',
            "result_path": '/home/thang/Desktop/Kolkata/T1/classification/T1_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/Kolkata/T1/cloud/T1.shp",
            "month": '01',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/thang/Desktop/Kolkata/T2/mosaic_statistic/T2_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/Kolkata/T2/mosaic_statistic/T2_DA.tif',
            "result_path": '/home/thang/Desktop/Kolkata/T2/classification/T2_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/Kolkata/T2/cloud/T2.shp",
            "month": '02',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/thang/Desktop/Kolkata/T3/mosaic_statistic/T3_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/Kolkata/T3/mosaic_statistic/T3_DA.tif',
            "result_path": '/home/thang/Desktop/Kolkata/T3/classification/T3_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/Kolkata/T3/cloud/T3.shp",
            "month": '03',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/thang/Desktop/Kolkata/T4/mosaic_statistic/T4_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/Kolkata/T4/mosaic_statistic/T4_DA.tif',
            "result_path": '/home/thang/Desktop/Kolkata/T4/classification/T4_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/Kolkata/T4/cloud/T4.shp",
            "month": '04',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/thang/Desktop/Kolkata/T5/mosaic_statistic/T5_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/Kolkata/T5/mosaic_statistic/T5_DA.tif',
            "result_path": '/home/thang/Desktop/Kolkata/T5/classification/T5_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/Kolkata/T5/cloud/T5.shp",
            "month": '05',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/thang/Desktop/Kolkata/T6/mosaic_statistic/T6_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/Kolkata/T6/mosaic_statistic/T6_DA.tif',
            "result_path": '/home/thang/Desktop/Kolkata/T6/classification/T6_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/Kolkata/T6/cloud/T6.shp",
            "month": '06',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/thang/Desktop/Kolkata/T7/mosaic_statistic/T7_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/Kolkata/T7/mosaic_statistic/T7_DA.tif',
            "result_path": '/home/thang/Desktop/Kolkata/T7/classification/T7_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/Kolkata/T7/cloud/T7.shp",
            "month": '07',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/thang/Desktop/Kolkata/T8/mosaic_statistic/T8_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/Kolkata/T8/mosaic_statistic/T8_DA.tif',
            "result_path": '/home/thang/Desktop/Kolkata/T8/classification/T8_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/Kolkata/T8/cloud/T8.shp",
            "month": '08',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/thang/Desktop/Kolkata/T9/mosaic_statistic/T9_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/Kolkata/T9/mosaic_statistic/T9_DA.tif',
            "result_path": '/home/thang/Desktop/Kolkata/T9/classification/T9_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/Kolkata/T9/cloud/T9.shp",
            "month": '09',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/thang/Desktop/Kolkata/T10/mosaic_statistic/T10_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/Kolkata/T10/mosaic_statistic/T10_DA.tif',
            "result_path": '/home/thang/Desktop/Kolkata/T10/classification/T10_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/Kolkata/T10/cloud/T10.shp",
            "month": '10',
            "year": 2021
        },
    ]
    #
    # for i, image in enumerate(sentinels_kolkata):
    #     before_month = sentinels[i - 1]['month'] if i > 0 else None
    #     main(image['statistic_mosaic_path'], image['view_mosaic_path'], image['result_path'], image['cloud_path'],
    #          'Sentinel', image['month'], before_month, image['year'], image['year'])

    sentinels_hydrabat = [
        {
            "statistic_mosaic_path": '/home/thang/Desktop/data_hyrabat/T1/mosaic_statistic/T1_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/data_hyrabat/T1/mosaic_statistic/T1_DA.tif',
            "result_path": '/home/thang/Desktop/data_hyrabat/T1/classification/T1_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/data_hyrabat/T1/cloud/T1.shp",
            "month": '01',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/thang/Desktop/data_hyrabat/T2/mosaic_statistic/T2_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/data_hyrabat/T2/mosaic_statistic/T2_DA.tif',
            "result_path": '/home/thang/Desktop/data_hyrabat/T2/classification/T2_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/data_hyrabat/T2/cloud/T2.shp",
            "month": '02',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/thang/Desktop/data_hyrabat/T3/mosaic_statistic/T3_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/data_hyrabat/T3/mosaic_statistic/T3_DA.tif',
            "result_path": '/home/thang/Desktop/data_hyrabat/T3/classification/T3_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/data_hyrabat/T3/cloud/T3.shp",
            "month": '03',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/thang/Desktop/data_hyrabat/T4/mosaic_statistic/T4_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/data_hyrabat/T4/mosaic_statistic/T4_DA.tif',
            "result_path": '/home/thang/Desktop/data_hyrabat/T4/classification/T4_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/data_hyrabat/T4/cloud/T4.shp",
            "month": '04',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/thang/Desktop/data_hyrabat/T5/mosaic_statistic/T5_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/data_hyrabat/T5/mosaic_statistic/T5_DA.tif',
            "result_path": '/home/thang/Desktop/data_hyrabat/T5/classification/T5_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/data_hyrabat/T5/cloud/T5.shp",
            "month": '05',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/thang/Desktop/data_hyrabat/T6/mosaic_statistic/T6_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/data_hyrabat/T6/mosaic_statistic/T6_DA.tif',
            "result_path": '/home/thang/Desktop/data_hyrabat/T6/classification/T6_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/data_hyrabat/T6/cloud/T6.shp",
            "month": '06',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/thang/Desktop/data_hyrabat/T7/mosaic_statistic/T7_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/data_hyrabat/T7/mosaic_statistic/T7_DA.tif',
            "result_path": '/home/thang/Desktop/data_hyrabat/T7/classification/T7_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/data_hyrabat/T7/cloud/T7.shp",
            "month": '07',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/thang/Desktop/data_hyrabat/T8/mosaic_statistic/T8_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/data_hyrabat/T8/mosaic_statistic/T8_DA.tif',
            "result_path": '/home/thang/Desktop/data_hyrabat/T8/classification/T8_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/data_hyrabat/T8/cloud/T8.shp",
            "month": '08',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/thang/Desktop/data_hyrabat/T9/mosaic_statistic/T9_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/data_hyrabat/T9/mosaic_statistic/T9_DA.tif',
            "result_path": '/home/thang/Desktop/data_hyrabat/T9/classification/T9_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/data_hyrabat/T9/cloud/T9.shp",
            "month": '09',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/thang/Desktop/data_hyrabat/T10/mosaic_statistic/T10_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/data_hyrabat/T10/mosaic_statistic/T10_DA.tif',
            "result_path": '/home/thang/Desktop/data_hyrabat/T10/classification/T10_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/data_hyrabat/T10/cloud/T10.shp",
            "month": '10',
            "year": 2021
        },
        {
            "statistic_mosaic_path": '/home/thang/Desktop/data_hyrabat/T11/mosaic_statistic/T11_DA.tif',
            "view_mosaic_path": '/home/thang/Desktop/data_hyrabat/T11/mosaic_statistic/T11_DA.tif',
            "result_path": '/home/thang/Desktop/data_hyrabat/T11/classification/T11_DA_color.tif',
            "cloud_path": "/home/thang/Desktop/data_hyrabat/T11/cloud/T11.shp",
            "month": '10',
            "year": 2021
        },
    ]
    # for i, image in enumerate(sentinels_hydrabat):
    #     before_month = sentinels[i - 1]['month'] if i > 0 else None
    #     main(image['statistic_mosaic_path'], image['view_mosaic_path'], image['result_path'], image['cloud_path'],
    #          'Sentinel', image['month'], before_month, image['year'], image['year'])

    #
    # for i, image in enumerate(jilins):
    #     before_month = jilins[i - 1]['month'] if i > 0 else None
    #     main(image['mosaic_path'],image['view_mosaic_path'], image['result_path'], None, 'Jilin', image['month'], before_month, image['year'],image['year'])

# def get_data_from_postgis_db(id, src, aoi, type, month, year):
#     fields = {'id': id, 'src_id': src['id'], 'aoi_id': aoi['iid'], type: type['id'], "month": month, "year": year}
#     engine = create_engine(SQLALCHEMY_DATABASE_URI)
#     src_image = Imagery.query().filter_by(id=image_id).first()
#
#     with engine.connect() as connection:
#         images = connection.execute(sql)
