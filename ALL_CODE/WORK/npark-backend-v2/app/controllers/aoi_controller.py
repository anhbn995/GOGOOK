import json
import os
import io
import shutil
from zipfile import ZipFile
from flask import request, send_file
import geopandas as gpd
from flask import request
from flask_restful import Resource
from geopandas import GeoSeries
import uuid
from app.utils.geometry import remove_third_dimension
from app.utils.path import make_temp_folder_in_root_data_folder, make_temp_folder
from app.utils.response import success, bad_request
from app.utils.path import path_2_abs
from PIL import Image
from app.utils.statistic import calculate_metadata
from config.default import ROOT_DATA_FOLDER
import rasterio
from app.utils.geospatial import get_geospatial_data


class AOIController(Resource):

    def convert_to_geojson(self):
        try:
            geom_dataframe = _get_geodataframe()
        except Exception:
            return bad_request()

        columns = geom_dataframe.columns.values.tolist()
        df = _remove_z_index(geom_dataframe)
        geom_dataframe['geometry'] = df.geometry.tolist()
        data = json.loads(geom_dataframe.to_json())
        data['properties'] = []
        for column in columns:
            if column != 'geometry':
                data['properties'].append({
                    'name': column,
                    'type': 'text'
                })
        return success(data)

    def convert_to_geojson_from_path(self):
        payload = request.args
        path = path_2_abs(payload.get("path"))
        geom_dataframe = gpd.read_file(path)
        columns = geom_dataframe.columns.values.tolist()
        df = _remove_z_index(geom_dataframe)
        geom_dataframe['geometry'] = df.geometry.tolist()
        data = json.loads(geom_dataframe.to_json())
        data['properties'] = []
        for column in columns:
            if column != 'geometry':
                data['properties'].append({
                    'name': column,
                    'type': 'text'
                })
        return success(data)

    def geojson_convert_crs(self):
        try:
            payload = json.loads(request.data)
            geojson = payload.get('geojson')
            temp_dir = make_temp_folder_in_root_data_folder()

            temp_file = f'{temp_dir}/input.geojson'
            with open(temp_file, 'w') as json_file:
                json.dump(geojson, json_file)
            geom_dataframe = gpd.read_file(temp_file)
            geom_dataframe = geom_dataframe.to_crs(3857)
            data = json.loads(geom_dataframe.to_json())
            shutil.rmtree(temp_dir)
            return success(data)
        except Exception:
            # shutil.rmtree(temp_dir)
            return bad_request()

    def save_geojson(self):
        # try:
        payload = json.loads(request.data)
        geojson = payload.get('geojson')
        temp_dir = make_temp_folder_in_root_data_folder()
        file_id = uuid.uuid4()
        temp_file_geojson = f'{temp_dir}/{file_id.hex}.geojson'
        temp_file_shp = f'{temp_dir}/{file_id.hex}.shp'
        with open(temp_file_geojson, 'w') as json_file:
            json.dump(geojson, json_file)
        df = gpd.read_file(temp_file_geojson)
        df.to_file(temp_file_shp, driver='ESRI Shapefile')
        os.remove(temp_file_geojson)
        data = {
            "abs_path": os.path.abspath(temp_file_shp),
            "file_name": file_id.hex
        }
        return success(data)

    # except Exception:
    #     # shutil.rmtree(temp_dir)
    #
    #     return bad_request()

    def create_yaml(self):

        payload = json.loads(request.data)
        data_yaml = payload.get('yaml')
        file_id = uuid.uuid4()
        temp_dir = make_temp_folder_in_root_data_folder()
        temp_file = f'{temp_dir}/{file_id.hex}.yaml'

        f = open(temp_file, "w")
        f.write(data_yaml)
        f.close()

        data = {
            "abs_path": os.path.abspath(temp_file)
        }
        return success(data)

    def create_single_color_image(self, color):
        hlen = len(color)
        color_rgb = tuple(int(color[i:i + hlen // 3], 16) for i in range(0, hlen, int(hlen // 3)))
        temp_dir = make_temp_folder()
        file_id = uuid.uuid4()
        temp_file = f'{temp_dir}/{file_id.hex}.png'
        img = Image.new('RGB', (200, 200), color_rgb)
        img.save(temp_file)
        return send_file(temp_file, mimetype='image/gif')

    def calculate_image(self):
        payload = request.args
        file_path = f"{payload.get('path')}"
        if not os.path.exists(file_path):
            file_path = f"{ROOT_DATA_FOLDER}{file_path}"
        if file_path.lower().endswith(('.tif')):
            data = {
                "meta": calculate_metadata(file_path),
                "nodata": _get_image_nodata(file_path)
            }
        else:
            geospatial_data = get_geospatial_data(file_path)
            data = {
                'srs': geospatial_data.get('srs'),
                'geometry': geospatial_data.get('bbox'),
            }
        return success(data)


def _get_image_nodata(path):
    with rasterio.open(path) as ds:
        return ds.nodata or 0


def _get_geodataframe():
    gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
    file_type = request.form.get("type")
    temp_dir = make_temp_folder()
    if file_type == 'zip':
        geom_dataframe = _get_geometry_zip(temp_dir)
    elif file_type == 'shp':
        geom_dataframe = _get_geometry_shp(temp_dir)
    elif file_type == 'kml':
        geom_dataframe = _get_geometry_kml(temp_dir)
    elif file_type == 'kmz':
        geom_dataframe = _get_geometry_kmz(temp_dir)
    elif file_type == 'gml':
        geom_dataframe = _get_geometry_gml(temp_dir)
    else:
        geom_dataframe = _get_geometry_geojson(temp_dir)

    from shapely.geometry.collection import GeometryCollection

    geom_dataframe['geometry'] = geom_dataframe.geometry.apply(lambda x: x if x else GeometryCollection())

    geom_dataframe = geom_dataframe.to_crs('EPSG:4326')
    geom_dataframe = _transform_dataframe(geom_dataframe)
    shutil.rmtree(temp_dir)
    return geom_dataframe


def _transform_dataframe(dataframe):
    try:
        return dataframe.explode()
    except:
        return dataframe


def _remove_z_index(geom_dataframe):
    geom_2d_list = []
    for geom_3d in geom_dataframe['geometry']:
        geom_2d = remove_third_dimension(geom_3d)
        geom_2d_list.append(geom_2d)
    df = gpd.GeoDataFrame(geometry=GeoSeries(geom_2d_list))

    return df


def _get_geometry_zip(working_dir):
    zipfile = request.files.get('zip')
    temp_zip = '{}/temp.zip'.format(working_dir)
    zipfile.save(temp_zip)
    with ZipFile(temp_zip) as zip:
        zip.extractall(working_dir)
        _, dir, files = next(os.walk(working_dir))
        tmp_dir = working_dir
        if len(dir) != 0:
            tmp_dir = os.path.join(working_dir, dir[0])
            _, _, files = next(os.walk(tmp_dir))
        for file in files:
            if 'shp' in file or 'kml' in file or 'json' in file or 'gml' in file:
                _path = os.path.join(tmp_dir, file)
                return _get_geometry(_path)
            elif 'kmz' in file:
                _path = os.path.join(tmp_dir, file)
                return _get_geometry_kmz(working_dir, _path)
    return gpd.read_file(f'zip://{temp_zip}')


def _get_geometry(path_file):
    return gpd.read_file(path_file)


def _get_geometry_shp(working_dir):
    prj = request.files.get('prj')
    shx = request.files.get('shx')
    shp = request.files.get('shp')
    dbf = request.files.get('dbf')
    qpj = request.files.get('qpj')
    if prj:
        temp_prj = '{}/temp.prj'.format(working_dir)
        prj.save(temp_prj)
    if qpj:
        temp_qpj = '{}/temp.qpj'.format(working_dir)
        qpj.save(temp_qpj)
    if shx:
        temp_shx = '{}/temp.shx'.format(working_dir)
        shx.save(temp_shx)
    if shp:
        temp_shp = '{}/temp.shp'.format(working_dir)
        shp.save(temp_shp)
    if dbf:
        temp_dbf = '{}/temp.dbf'.format(working_dir)
        dbf.save(temp_dbf)
    return gpd.read_file(temp_shp)


def _get_geometry_kml(working_dir):
    kml = request.files.get('kml')
    temp_kml = '{}/temp.kml'.format(working_dir)
    kml.save(temp_kml)
    return gpd.read_file(temp_kml, driver='KML')


def _get_geometry_gml(working_dir):
    gml = request.files.get('gml')
    temp_gml = '{}/temp.gml'.format(working_dir)
    gml.save(temp_gml)
    return gpd.read_file(temp_gml, driver='GML')


def _get_geometry_kmz(working_dir):
    kmz = request.files.get('kmz')
    temp_kmz = '{}/temp.kmz'.format(working_dir)
    kmz.save(temp_kmz)
    kmz = ZipFile(temp_kmz, 'r')
    kmz.extract('doc.kml', working_dir)
    return gpd.read_file(os.path.join(working_dir, 'doc.kml'), driver='KML')


def _get_geometry_geojson(working_dir):
    geojson = request.files.get('geojson')
    temp_geojson = '{}/temp.geojson'.format(working_dir)
    geojson.save(temp_geojson)
    return gpd.read_file(temp_geojson, driver='GeoJSON')
