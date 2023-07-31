import shutil

import rasterio
import cv2
import uuid
from osgeo import gdal, osr, ogr
import os
import json
from app.utils.path import make_temp_folder
from pathlib import Path
import numpy as np
from lib import ogr2ogr
from config.default import ROOT_DATA_FOLDER
import geopy.distance
from sqlalchemy import create_engine
from config.default import SQLALCHEMY_DATABASE_URI
import ast
import geopandas as gpd
from import_vectors import store_vectors
from app.utils import get_file_size
import requests
from config.default import HOSTED_ENDPOINT
from app.utils import statistic
from inject_image import _translate

GDAL_OGR_TYPE_MAPPER = {
    gdal.GDT_Byte: ogr.OFTInteger,
    gdal.GDT_UInt16: ogr.OFTInteger,
    gdal.GDT_Int16: ogr.OFTInteger,
    gdal.GDT_UInt32: ogr.OFTInteger,
    gdal.GDT_Int32: ogr.OFTInteger,
    gdal.GDT_Float32: ogr.OFTReal,
    gdal.GDT_Float64: ogr.OFTReal,
    gdal.GDT_CInt16: ogr.OFTInteger,
    gdal.GDT_CInt32: ogr.OFTInteger,
    gdal.GDT_CFloat32: ogr.OFTReal,
    gdal.GDT_CFloat64: ogr.OFTReal
}


def vectorize_dataset(input_tif, output_geojson, labels, epsilon, selected_values):
    with rasterio.open(input_tif, 'r', driver='GTiff') as src:
        img = src.read(1)
        crs = dict(src.crs)
        transform = src.transform
        w, h = src.width, src.height
        num_bands = src.count
        transform = src.transform

    dst = gdal.Open(input_tif)
    gt = dst.GetGeoTransform()
    coords_origin = (gt[3], gt[0])
    coords_right = (gt[3], gt[0] + gt[1])
    # import pdb
    # pdb.set_trace()

    pixcel_size = geopy.distance.geodesic(coords_origin, coords_right).km * 1000

    ele_value = round(epsilon / abs(pixcel_size))
    print(ele_value)
    ele_value = 1 if ele_value == 0 else ele_value
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (ele_value, ele_value))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ele_value, ele_value))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ele_value, ele_value))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (ele_value, ele_value))

    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel3)
    for i in range(5):
        closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel2)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
    del closing
    for i in range(5):
        opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel2)

    image = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel3)

    mask_ = 1 * (image > 1)

    for value in selected_values:
        mask_ = mask_ + (1 * (image == value))

    mask_bool = mask_ > 1
    masked = 1 * mask_bool
    mask_final = masked * image

    mask_int = np.array(mask_final).astype('uint8')
    folder = Path(input_tif).parent
    str_id = uuid.uuid4().hex
    tmp_tif_path = '{}/{}.tif'.format(folder, str_id)
    tmp_geojson_path = '{}/{}.geojson'.format(folder, str_id)

    result = rasterio.open(tmp_tif_path, 'w', driver='GTiff',
                           height=h, width=w,
                           count=num_bands, dtype='uint8',
                           crs=crs,
                           transform=transform,
                           compress='lzw')
    result.write(mask_int, 1)
    result.close()

    polygonize(tmp_tif_path, tmp_geojson_path)
    ogr2ogr.main(["", "-f", "geojson", '-t_srs', 'epsg:4326', output_geojson, tmp_geojson_path])
    os.remove(tmp_tif_path)
    os.remove(tmp_geojson_path)
    print("tmp_tif_path", tmp_tif_path)
    print("tmp_geojson_path", tmp_geojson_path)

    with open(output_geojson) as outfile:
        data = json.load(outfile)

    def find(arr, el):
        for e in arr:
            if int(e['value']) == int(el):
                return e

    selected_labels = []
    for value in selected_values:
        selected_labels.append(find(labels, value))

    data['labels'] = selected_labels
    return data


def polygonize(img, shp_path):
    ds = gdal.Open(img)
    prj = ds.GetProjection()
    srcband = ds.GetRasterBand(1)
    dst_layername = "Shape"
    drv = ogr.GetDriverByName("GeoJSON")
    dst_ds = drv.CreateDataSource(shp_path)
    srs = osr.SpatialReference(wkt=prj)

    dst_layer = dst_ds.CreateLayer(dst_layername, srs=srs)
    raster_field = ogr.FieldDefn('raster_val', GDAL_OGR_TYPE_MAPPER[srcband.DataType])

    raster_field_1 = ogr.FieldDefn('label', GDAL_OGR_TYPE_MAPPER[srcband.DataType])
    dst_layer.CreateField(raster_field)
    dst_layer.CreateField(raster_field_1)
    gdal.Polygonize(srcband, srcband, dst_layer, 0, [], callback=None)
    del img, ds, srcband, dst_ds, dst_layer


def reproject_image_goc(src_path, dst_path, dst_crs='EPSG:4326'):
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


def reproject_image(src_path, dst_path, dst_crs='EPSG:4326'):
    with rasterio.open(src_path) as ds:
        nodata = ds.nodata or 0
    if ds.crs.to_string() != dst_crs:
        print(f'convert to {dst_crs}')
        temp_path = dst_path.replace('.tif', 'temp.tif')
        option = gdal.TranslateOptions(gdal.ParseCommandLine("-co \"TFW=YES\""))
        gdal.Translate(temp_path, src_path, options=option)
        option = gdal.WarpOptions(gdal.ParseCommandLine("-t_srs {} -dstnodata {}".format(dst_crs, nodata)))
        gdal.Warp(dst_path, temp_path, options=option)
        os.remove(temp_path)
    else:
        print(f'coppy image to {dst_path}')
        shutil.copyfile(src_path, dst_path)
        print('done coppy')
    return True


def rasterize_green_cover(list_images, temp_folder, labels, type='green_cover', name='green_cover'):
    """DUC ANH XONG XOA"""
    # print("DUC ANH XONG XOA")
    # path_img_nodata = r"/home/skm/SKM16/Work/Npark_planet2/Xoa_tren_serve.tif"
    # with rasterio.open(path_img_nodata) as src_mask:
    #     mask_nodata_t3 = src_mask.read(1)

    for index, image_from_db in enumerate(list_images):
        if not os.path.exists(f'{image_from_db["path"]}'):
            classification_image_path = f'{ROOT_DATA_FOLDER}{image_from_db["path"]}'
        else:
            classification_image_path = f'{image_from_db["path"]}'

        year = image_from_db['year']
        month = image_from_db['month']

        dir_path_vector = f'{ROOT_DATA_FOLDER}/data/{year}/{month}/vectors'
        if not os.path.exists(dir_path_vector):
            os.makedirs(dir_path_vector)

        fid_result = uuid.uuid4()
        result_geojson = f'{dir_path_vector}/{fid_result.hex}.geojson'

        reclassification_path = f'{temp_folder}/reclassification_mosaic111.tif'
        with rasterio.open(classification_image_path) as dst:
            band = dst.read(1)#*mask_nodata_t3
            meta = dst.meta

        with rasterio.open(reclassification_path, 'w', **meta) as dst1:
            dst1.write(band + 1, 1)

        vectorize_dataset(reclassification_path, result_geojson, labels, 31, [2])
        df = gpd.read_file(result_geojson)
        # import pdb
        # pdb.set_trace()
        store_vectors([{
            'id': fid_result.hex,
            'name': f'{year}_{month}_{name}',
            'path': f"/{os.path.relpath(result_geojson, ROOT_DATA_FOLDER)}",
            'month': month,
            'year': year,
            'type': type,
            'image_id': image_from_db["id"].hex,
            # 'geometry': ast.literal_eval(df.to_json()),
            'geometry': {},
            'src': 'Sentinel',
            'aoi_id': image_from_db["aoi_id"]
        }])


def largest_subarray(A, B):
    dims = np.minimum(A.shape, B.shape)  # find smallest dimensions
    idx = tuple(slice(None, dd) for dd in dims)  # construct tuple of slice indices
    return A[idx], B[idx]


def store_results(payload):
    url = '{}/internal/imageries'.format(
        HOSTED_ENDPOINT)
    print(url)
    r = requests.post(url, json=payload)

    if not r.ok:
        print(r)

        raise Exception('Fail to upload result')


def on_success(images):
    payload = []
    for image in images:
        meta = statistic.calculate_metadata(image['path'])
        if "properties" in meta and type(meta['properties']) is str:
            meta['properties'] = ast.literal_eval(meta['properties'])
        tile_url = '/v2/tile/{}/{}/{}'.format(image['year'], image['month'], image['file_id'].hex) + '/{z}/{x}/{y}'
        if "properties" in image:
            meta['properties'] = image['properties']
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


def vectorize_image(temp_folder, src, month, month_before, year, before_year, initial_aoi):
    """DUC ANH XONG XOA"""
    # path_img_nodata = r"/home/skm/SKM16/Work/Npark_planet2/Xoa_tren_serve.tif"
    # with rasterio.open(path_img_nodata) as src_mask:
    #     mask_nodata_t3 = src_mask.read(1)

    aoi_id = initial_aoi["id"]
    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    sql = f"select id, name, path, month, year,aoi_id from images where src_id={src['id']} and type_id=1 " \
          f" and aoi_id ={aoi_id} and ((month ='{month}' and year ={year}) or (month ='{month_before}'and year ={before_year})) order by created_at"

    sql2 = f"select id, name, path, month, year,aoi_id from images where src_id={src['id']} and type_id=8 " \
           f" and aoi_id ={aoi_id} and ((month ='{month}' and year ={year}) or (month ='{month_before}'and year ={before_year})) order by created_at"
    # order by year desc, month desc
    with engine.connect() as connection:
        images = connection.execute(sql)
        forest_images = connection.execute(sql2)

    list_images_db = []
    for index, image in enumerate(images):
        id1, name1, path1, month1, year1, aoi_id = image
        list_images_db.append({
            'month': month1,
            'year': year1,
            'name': name1,
            'path': path1,
            'id': id1,
            'aoi_id': aoi_id
        })
    list_forest_images_db = []
    for index, image in enumerate(forest_images):
        id1, name1, path1, month1, year1, aoi_id = image
        list_forest_images_db.append({
            'month': month1,
            'year': year1,
            'name': name1,
            'path': path1,
            'id': id1,
            'aoi_id': aoi_id
        })
    print(list_forest_images_db, "NDA1")
    payload = []

    # vectorize green cover
    labels = [{"color": "#228b22", "value": 2, "label": "Green cover"},
              {"color": "#11b9db", "value": 3, "label": "Water"},
              {"color": "#e47a10", "value": 4, "label": "Non-green"}]
    print(list_images_db, "NDA")
    rasterize_green_cover([list_images_db[1]], temp_folder, labels, type='green_cover', name='green_cover')

    # change green cover vector
    """green cover change"""
    list_images = []

    for index, image_from_db in enumerate(list_images_db):
        if index > 0:
            path1 = f'{ROOT_DATA_FOLDER}{list_images_db[index - 1]["path"]}'
            path2 = f'{ROOT_DATA_FOLDER}{image_from_db["path"]}'
            year = image_from_db["year"]
            month = image_from_db["month"]
            aoi_id = image_from_db["aoi_id"]

            with rasterio.open(path1) as dst1:
                band1 = dst1.read(1)#*mask_nodata_t3
                meta = dst1.meta
            with rasterio.open(path2) as dst2:
                band2 = dst2.read(1)#*mask_nodata_t3
            band1, band2 = largest_subarray(band1, band2)
            band = np.where((band1 == 1) & (band2 != 1), 2, 0).astype('uint8')
            band[(band1 != 1) & (band2 == 1)] = 3

            dir_path_image = f'{ROOT_DATA_FOLDER}/data/{image_from_db["year"]}/{image_from_db["month"]}/images'
            fid_image = uuid.uuid4()

            pre_translate_path = f'{temp_folder}/{fid_image.hex}.tif'
            path = f'{dir_path_image}/{fid_image.hex}.tif'
            meta.update({"dtype": 'uint8'})
            with rasterio.open(pre_translate_path, 'w', **meta) as dst:
                dst.write(band, 1)
            _translate(pre_translate_path, path)
            list_images.append({
                'file_id': fid_image,
                'name': f'{year}_{month}_green_cover_change',
                'path': path,
                'month': month,
                'year': year,
                'properties': {
                    "mode": "unique", "bands": [0, 0, 0], "contrastEnhancement": "no",
                    "listValue": [
                        {"color": "#b9b408", "minValue": 2, "name": "Green to Non"},
                        {"color": "#3FF980", "minValue": 3, "name": "Non to Green"}
                    ],
                },
                'src': src['key'],
                'type': 'green_cover_change',
                'aoi_id': aoi_id
            })

            temp_reproject = f'{temp_folder}/4326_mosaic.tif'
            reproject_image(path, temp_reproject)

            labels = [{"color": "#228b22", "value": 2, "label": "Green to Non"},
                      {"color": "#11b9db", "value": 3, "label": "Non to Green"}]

            dir_path_vector = f'{ROOT_DATA_FOLDER}/data/{image_from_db["year"]}/{image_from_db["month"]}/vectors'
            if not os.path.exists(dir_path_vector):
                os.makedirs(dir_path_vector)
            fid = uuid.uuid4()
            temp_path_vector = f'{temp_folder}/{fid.hex}.geojson'

            vectorize_dataset(temp_reproject, temp_path_vector, labels, 14, [2, 3])
            df = gpd.read_file(temp_path_vector)
            df['type'] = df['raster_val'].map(lambda x: 'Green to Non' if x == 2 else 'Non to Green')
            abs_path_vector = f'{dir_path_vector}/{fid.hex}.geojson'
            df.to_file(abs_path_vector, driver='GeoJSON')
            payload.append({
                'id': str(fid),
                'name': f'{image_from_db["year"]}_{image_from_db["month"]}_green_cover_change',
                'path': f"/{os.path.relpath(abs_path_vector, ROOT_DATA_FOLDER)}",
                'month': image_from_db["month"],
                'year': image_from_db["year"],
                'type': 'green_cover_change',
                'geometry': {},
                'image_id': image_from_db["id"].hex,
                'src': 'Sentinel',
                "aoi_id": aoi_id
            })

    # change forest cover vector
    """forest cover change"""
    list_temp_reclass = []
    for forest_path in list_forest_images_db:
        temp_reclass = f'{temp_folder}_temp_reclass'
        temp_object_path = forest_path.copy()
        with rasterio.open(f'{ROOT_DATA_FOLDER}/{forest_path["path"]}') as dst:
            band1 = dst.read(1)#*mask_nodata_t3
            meta = dst.meta
            band1[band1 > 0] = 1
        with rasterio.open(temp_reclass, 'w', **meta) as dst:
            dst.write(band1, 1)
        temp_object_path['path'] = temp_reclass
        list_temp_reclass.append(temp_object_path)
    labels = [{"color": "#4CAF50", "value": 2, "name": "Forest"}]

    rasterize_green_cover(list_temp_reclass, temp_folder, labels, type='forest_cover_area', name='forest_area')
    for index, image_from_db in enumerate(list_forest_images_db):
        if index > 0:

            path1 = f'{ROOT_DATA_FOLDER}{list_forest_images_db[index - 1]["path"]}'
            path2 = f'{ROOT_DATA_FOLDER}{image_from_db["path"]}'
            year = image_from_db["year"]
            month = image_from_db["month"]
            aoi_id = image_from_db["aoi_id"]

            with rasterio.open(path1) as dst1:
                band1 = dst1.read(1)#*mask_nodata_t3
                meta = dst1.meta
            with rasterio.open(path2) as dst2:
                band2 = dst2.read(1)#*mask_nodata_t3
            band1, band2 = largest_subarray(band1, band2)
            band = np.where((band1 >= 0.4) & (band2 <= 0.4), 2, 0).astype('uint8')
            band[(band1 < 0.4) & (band2 >= 0.4)] = 3

            dir_path_image = f'{ROOT_DATA_FOLDER}/data/{image_from_db["year"]}/{image_from_db["month"]}/images'
            fid_image = uuid.uuid4()

            pre_translate_path = f'{temp_folder}/{fid_image.hex}.tif'
            path = f'{dir_path_image}/{fid_image.hex}.tif'
            meta.update({"dtype": 'uint8'})
            with rasterio.open(pre_translate_path, 'w', **meta) as dst:
                dst.write(band, 1)
            _translate(pre_translate_path, path)
            list_images.append({
                'file_id': fid_image,
                'name': f'{year}_{month}_forest_area_change',
                'path': path,
                'month': month,
                'year': year,
                'properties': {
                    "mode": "unique", "bands": [0, 0, 0], "contrastEnhancement": "no",
                    "listValue": [
                        {"color": "#b9b408", "minValue": 2, "name": "Forest to Non"},
                        {"color": "#3FF980", "minValue": 3, "name": "Non to Forest"}
                    ],
                },
                'src': src['key'],
                'type': 'forest_cover_change',
                'aoi_id': aoi_id
            })

            temp_reproject = f'{temp_folder}/4326_mosaic.tif'
            reproject_image(path, temp_reproject)

            labels = [{"color": "#228b22", "value": 2, "label": "Forest to Non"},
                      {"color": "#11b9db", "value": 3, "label": "Non to Forest"}]

            dir_path_vector = f'{ROOT_DATA_FOLDER}/data/{image_from_db["year"]}/{image_from_db["month"]}/vectors'
            if not os.path.exists(dir_path_vector):
                os.makedirs(dir_path_vector)
            fid = uuid.uuid4()
            temp_path_vector = f'{temp_folder}/{fid.hex}.geojson'

            vectorize_dataset(temp_reproject, temp_path_vector, labels, 14, [2, 3])
            df = gpd.read_file(temp_path_vector)
            df['type'] = df['raster_val'].map(lambda x: 'Green to Non' if x == 2 else 'Non to Green')
            abs_path_vector = f'{dir_path_vector}/{fid.hex}.geojson'
            df.to_file(abs_path_vector, driver='GeoJSON')
            payload.append({
                'id': str(fid),
                'name': f'{image_from_db["year"]}_{image_from_db["month"]}_forest_cover_change',
                'path': f"/{os.path.relpath(abs_path_vector, ROOT_DATA_FOLDER)}",
                'month': image_from_db["month"],
                'year': image_from_db["year"],
                'type': 'forest_cover_change',
                'geometry': {},
                'image_id': image_from_db["id"].hex,
                'src': 'Sentinel',
                "aoi_id": aoi_id
            })

    try:
        on_success(list_images)

        for vector in payload:
            store_vectors([vector])
    except Exception as e:
        for image in list_images:
            os.remove(image["path"])

        for payload in payload:
            os.remove(f'{ROOT_DATA_FOLDER}{payload["path"]}')
        raise e


if __name__ == '__main__':
    temp_folder = make_temp_folder()
    source_image = {
        'id': 1,
        'key': 'Sentinel',
        'band_red': 3
    }
    aoi = {
        "id": 14
    }

    # vectorize_image(temp_folder, source_image, '11', '10', 2021, 2021, aoi)
    # vectorize_image(temp_folder, source_image, '10', '09', 2021, 2021, aoi)
    #
    # vectorize_image(temp_folder, source_image, '08', '07', 2021, 2021, aoi)
    # vectorize_image(temp_folder, source_image, '07', '06', 2021, 2021, aoi)
    # vectorize_image(temp_folder, source_image, '06', '05', 2021, 2021, aoi)
    # vectorize_image(temp_folder, source_image, '05', '04', 2021, 2021, aoi)
    # vectorize_image(temp_folder, source_image, '04', '03', 2021, 2021, aoi)
    # vectorize_image(temp_folder, source_image, '03', '02', 2021, 2021, aoi)
    # vectorize_image(temp_folder, source_image, '01', '01', 2021, 2021, aoi)
    for thang in range(1, 12):
        thang_truoc = f"{thang:02d}"
        thang_sau = f"{thang:02d}"
        vectorize_image(temp_folder, source_image, thang_sau, thang_truoc, 2021, 2021, aoi)
