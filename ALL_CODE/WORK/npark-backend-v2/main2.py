from app.utils.path import make_temp_folder
import uuid
import geopandas as gpd
import json
from sqlalchemy import create_engine
from config.default import SQLALCHEMY_DATABASE_URI
from clip_image import clip
import shutil
from import_images_v2 import remove_invalid_gdf, check_intersection, store_mosaic_image, reclass_image, \
    calc_vegetable_index, on_success, get_pixel_size, calc_area
from vectorize import vectorize_image
from import_vectors import store_cloud
# from app.models.imagery import Imagery
# import datetime
# from dateutil.relativedelta import relativedelta
# import os
from ultils import reproject_image

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
        'band_red': [6, 8]
    },
    {
        'id': 3,
        'key': 'Jilin',
        'band_red': 3
    }
]


# def reproject_image(src_path, dst_path, dst_crs='EPSG:4326'):
#     from osgeo import gdal
#     import rasterio
#     with rasterio.open(src_path) as ds:
#         nodata = ds.nodata or 0
#     temp_path = dst_path.replace('.tif', 'temp.tif')
#     option = gdal.TranslateOptions(gdal.ParseCommandLine("-co \"TFW=YES\""))
#     gdal.Translate(temp_path, src_path, options=option)
#     option = gdal.WarpOptions(gdal.ParseCommandLine("-t_srs {} -dstnodata {}".format(dst_crs, nodata)))
#     gdal.Warp(dst_path, temp_path, options=option)
#     os.remove(temp_path)
#     return True


def main(image_mosaic_statistic_path: str, image_mosaic_view_path: str, image_classification_path: str,
         image_forest_path: str, cloud_path: str,
         src: str, month: str, before_month, year: int, before_year):
    source_image = next((item for item in SOURCE_TYPES if item["key"] == src), None)
    band_red = source_image['band_red']
    temp_folder = make_temp_folder()
    print(SQLALCHEMY_DATABASE_URI)
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

            # mosaic_image_statistic_reproject = f'{temp_folder}/mosaic_image_statistic_reproject.tif'
            # reproject_image(image_mosaic_statistic_path, mosaic_image_statistic_reproject)
            # image_mosaic_statistic_path = mosaic_image_statistic_reproject

            forest_image_reproject = f'{temp_folder}/forest_image_reproject.tif'
            reproject_image(image_forest_path, forest_image_reproject)
            # image_forest_path = forest_image_reproject

            list_images = []
            init_mosaic_image = image_mosaic_statistic_path
            init_image = image_classification_path
            init_forest_image = image_forest_path

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

            forest_image = f'{temp_folder}/forest_crop.tif'
            clip(init_forest_image, aoi['geometry'], forest_image, temp_folder)

            print('***'*50)

            fid_mosaic = uuid.uuid4()
            list_image = store_mosaic_image(fid_mosaic, mosaic_image_statistic_crop, mosaic_image_view_crop,
                                            classification_image, year,
                                            month, src,
                                            initial_aoi, temp_folder)
            list_images += list_image
            reclass_image_path = f'{temp_folder}/reclass_mosaic.tif'
            print('1*'*20)
            reclass_image(mosaic_image_statistic_crop, classification_image, reclass_image_path, temp_folder)
            print('2*'*20)
            list_images += calc_vegetable_index(reclass_image_path, src, temp_folder, initial_aoi, month, year,
                                                classification_image, forest_image, band_red)
            print('3*'*20)
            print(list_images)
            on_success(list_images)

            print("Done store mosaic, classification, vegetable index")
            x_size, y_size = get_pixel_size(mosaic_image_statistic_crop)
            print(x_size, y_size)
            # print('4*'*20)
            calc_area(fid_mosaic, reclass_image_path, forest_image, x_size, y_size, band_red)
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


# def update_or_create_image(image_mosaic_statistic_path: str, image_mosaic_view_path: str,
#                            image_classification_path: str, cloud_path: str,
#                            start_date: str, end_date: str, src='Sentinel'):
#     start = datetime.strptime(start_date, '%d-%m-%Y')
#     month = start.month
#     year = start.year
#     main(image_mosaic_statistic_path, image_mosaic_view_path, image_classification_path, cloud_path, src, month,
#          before_month, year, before_year)


if __name__ == '__main__':
    # years = [2021, 2022]
    # for year in years:
    #     for month in range(1, 13):
    #         thang_nam = {
    #             "statistic_mosaic_path": f'/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/{year}_T{month}.tif',
    #             "view_mosaic_path": f'/home/geoai/geoai_data_test2/data_npark/Sentinel/mosaic/{year}_T{month}.tif',
    #             "result_path": f'/home/geoai/geoai_data_test2/data_npark/Sentinel/classification/{year}_T{month}.tif',
    #             "forest_path": f'/home/geoai/geoai_data_test2/data_npark/Sentinel/forest/T{month}_{year}_DA_forest2.tif',
    #             "cloud_path": f"/home/geoai/geoai_data_test2/data_npark/Sentinel/cloud/{year}_T{month}.shp",
    #             "month": '{:02}'.format(month),
    #             "year": year
    #         }
    #         if month == 1 and year == 2021:
    #             month_curent = '{:02}'.format(month)
    #             month_before = None
    #             year_curent = year
    #             year_before = year
    #
    #         elif month == 1 and year == 2022:
    #             month_curent = '{:02}'.format(month)
    #             month_before = '{:02}'.format(12)
    #             year_curent = year
    #             year_before = year - 1
    #         elif month == 4 and year == 2022:
    #             break
    #         else:
    #             month_curent = '{:02}'.format(month)
    #             month_before = '{:02}'.format(month - 1)
    #             year_curent = year
    #             year_before = year
    #         print(month_curent, year_curent, month_before, year_before)
    #         main(thang_nam['statistic_mosaic_path'], thang_nam['view_mosaic_path'], thang_nam['result_path'],
    #              thang_nam['forest_path'], thang_nam['cloud_path'], 'Sentinel',
    #              month_curent, month_before, year_curent, year_before)
    year = 2022
    months = [7]
    for month in months:
        thang_nam = {
            "statistic_mosaic_path": f'/home/skm/SKM16/IMAGE/npark/npark-backend-v2/data_projects/Green Cover Npark Singapore/results/mosaic/T{month}-{year}.tif',
            "view_mosaic_path": f'/home/skm/SKM16/IMAGE/npark/npark-backend-v2/data_projects/Green Cover Npark Singapore/results/view/T{month}-{year}.tif',
            "result_path": f'/home/skm/SKM16/IMAGE/npark/npark-backend-v2/data_projects/Green Cover Npark Singapore/results/classification/T{month}-{year}.tif',
            "forest_path": f'/home/skm/SKM16/IMAGE/npark/npark-backend-v2/data_projects/Green Cover Npark Singapore/results/forest/T{month}-{year}.tif',
            "cloud_path": f"/home/skm/SKM16/IMAGE/npark/npark-backend-v2/data_projects/Green Cover Npark Singapore/results/cloud/T{month}-{year}.shp",
            "month": '{:02}'.format(month),
            "year": year
        }
        main(thang_nam['statistic_mosaic_path'], thang_nam['view_mosaic_path'], thang_nam['result_path'],
             thang_nam['forest_path'], thang_nam['cloud_path'], 'Sentinel',
             '{:02}'.format(month), '{:02}'.format(month - 1), year, year)
    print("NDASDASDASDASD")

    #
    # year = 2022
    # months = [8]
    # for month in months:
    #     thang_nam = {
    #         "statistic_mosaic_path": f'/home/skm/SKM16/IMAGE/npark/npark-backend-v2/data_projects/Green Cover Npark Singapore/PlanetScope/v_4326/mosaic/T{month-1}_{year}.tif',
    #         "view_mosaic_path": f'/home/skm/SKM16/IMAGE/npark/npark-backend-v2/data_projects/Green Cover Npark Singapore/PlanetScope/v_4326/view/T{month-1}_{year}.tif',
    #         "result_path": f'/home/skm/SKM16/IMAGE/npark/npark-backend-v2/data_projects/Green Cover Npark Singapore/PlanetScope/v_4326/classification/T{month-1}_{year}.tif',
    #         "forest_path": f'/home/skm/SKM16/IMAGE/npark/npark-backend-v2/data_projects/Green Cover Npark Singapore/PlanetScope/v_4326/forest/T{month-1}_{year}.tif',
    #         "cloud_path": f"/home/skm/SKM16/IMAGE/npark/npark-backend-v2/data_projects/Green Cover Npark Singapore/PlanetScope/v_4326/cloud/T{month-1}_{year}.shp",
    #         "month": '{:02}'.format(month),
    #         "year": year
    #     }
    #     main(thang_nam['statistic_mosaic_path'], thang_nam['view_mosaic_path'], thang_nam['result_path'],
    #          thang_nam['forest_path'], None, 'PlanetScope',
    #          '{:02}'.format(month), None, year, year)#thang_nam['cloud_path'], '{:02}'.format(month - 1)
    # print("NDASDASDASDASD")

    # month = 3
    # thang_nam = {
    #             "statistic_mosaic_path": f'/home/skm/SKM16/Work/Npark_planet2/img_uint8/T{month}_{year}_4326.tif',
    #             "view_mosaic_path": f'/home/skm/SKM16/Work/Npark_planet2/img_uint8/RS_OK/view/T{month}_{year}_4326.tif',
    #             "result_path": f'/home/skm/SKM16/Work/Npark_planet2/img_uint8/RS_OK/T{month}_{year}_4326_color.tif',
    #             "forest_path": f'/home/skm/SKM16/Work/Npark_planet2/img_uint8/RS_OK/T{month}_{year}_4326_forest_oke_ok.tif',
    #             "cloud_path": f"/home/skm/SKM16/Work/Npark_planet2/img_uint8/RS_OK/T{month}_{year}_4326.shp",
    #             "month": '{:02}'.format(month),
    #             "year": year
    #         }
    # main(thang_nam['statistic_mosaic_path'], thang_nam['view_mosaic_path'], thang_nam['result_path'],
    #              thang_nam['forest_path'], None, 'PlanetScope',
    #              '{:02}'.format(month), None, year, year)#thang_nam['cloud_path'], '{:02}'.format(month - 1)

    # month4 = 7
    # thang_nam = {
    #             "statistic_mosaic_path": f'/home/skm/SKM16/Work/Npark_planet2/img_uint8/T{month4}_{year}_4326.tif',
    #             "view_mosaic_path": f'/home/skm/SKM16/Work/Npark_planet2/img_uint8/RS_OK/view/T{month4}_{year}_4326.tif',
    #             "result_path": f'/home/skm/SKM16/Work/Npark_planet2/img_uint8/RS_OK/T{month4}_{year}_4326_color.tif',
    #             "forest_path": f'/home/skm/SKM16/Work/Npark_planet2/img_uint8/RS_OK/T{month4}_{year}_4326_forest_oke_ok.tif',
    #             "cloud_path": f"/home/skm/SKM16/Work/Npark_planet2/img_uint8/RS_OK/T{month4}_{year}_4326.shp",
    #             "month": '{:02}'.format(month4),
    #             "year": year
    #         }
    # main(thang_nam['statistic_mosaic_path'], thang_nam['view_mosaic_path'], thang_nam['result_path'],
    #              thang_nam['forest_path'], None, 'PlanetScope',
    #              '{:02}'.format(month4), '{:02}'.format(month4 - 1), year, year)#thang_nam['cloud_path'], '{:02}'.format(month - 1)


    print("NDASDASDASDASD")