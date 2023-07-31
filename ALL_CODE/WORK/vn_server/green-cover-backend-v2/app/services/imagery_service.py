import uuid
import numpy as np
import json
import os
from app.utils.path import path_2_abs
from flask import send_from_directory
from osgeo import gdal
from shutil import rmtree
import geopandas as gpd
from rio_tiler.io import COGReader
from app.models.imagery import Imagery
from app.utils.response import success, not_found
from app.utils.string import get_unique_id
from app.utils.path import make_temp_folder
from app.utils.imagery import clip_image_with_aois, rgb_2_hex
from app.models.workspace import get_workspace_dir
import rasterio


# from app.microservices.gxl import gxl_rpcClient

class ColorValue:
    color = None
    value = None

    def __init__(self, color, value):
        self.color = color
        self.value = float(value)


class ImageryService:

    def clip_image_without_task(self, user_id, wks_id, image_id, aois):
        from config.default import HOSTED_ENDPOINT
        from app.utils import get_file_size
        import requests
        src_image = Imagery.query().filter_by(id=image_id).first()
        owner_id = src_image.owner_id
        wks_dir = get_workspace_dir(owner_id, wks_id)
        if not src_image:
            not_found()
        result_ids = []
        for _ in range(len(aois)):
            result_ids.append(get_unique_id())
        temp_dir = make_temp_folder()
        result_data = clip_image_with_aois(aois, image_id, result_ids, wks_dir, temp_dir, on_success=None,
                                           on_processing=None)
        for i, image_data in enumerate(result_data):
            tile_url = '/tile/{}/{}/{}'.format(user_id, uuid.UUID(str(wks_id)).hex,
                                               uuid.UUID(image_data['id']).hex) + '/{z}/{x}/{y}'
            metadata = src_image.meta
            metadata['geometry'] = image_data['geom']
            clipped_image_name = 'tmp_clipped_{}_{}'.format(i, src_image.name)

            payload = {
                'obj': {
                    'id': str(image_data['id']),
                    'created_by': user_id,
                    'owner_id': owner_id,
                    'workspace_id': wks_id,
                    'name': clipped_image_name,
                    'path': image_data['path'],
                    'tile_url': tile_url,
                    'meta': metadata,
                    'x_size': metadata.get('X_SIZE', 0),
                    'y_size': metadata.get('Y_SIZE', 0),
                    'size': get_file_size(image_data['path']),
                    'bbox': image_data['bbox']
                }
            }

            payload['result_type'] = 'image'
            payload['created_by'] = user_id
            url = '{}/internal/store_image'.format(HOSTED_ENDPOINT)
            requests.post(url, json=payload)

        response = {
            "data": result_ids,
            "message": "Clip image successful"
        }
        rmtree(temp_dir)
        return success(response)

    def send_image_to_client(self, user_id, workspace_id, imagery_id, name, type='tif'):
        if not name:
            imagery = Imagery.query().filter_by(workspace_id=workspace_id, id=imagery_id).first()
            if not imagery: not_found()
            user_id = imagery.owner_id
            name = imagery.name

        data_dir = get_workspace_dir(user_id, workspace_id)
        return send_from_directory(
            data_dir,
            "{}.{}".format(uuid.UUID(str(imagery_id)).hex, type),
            attachment_filename="{}.{}".format(name, type), as_attachment=True)

    def inspect_pixel_timeseries(self, wks_id, image_id, lat, lng):
        image = Imagery.query().filter_by(id=image_id).first()
        if not image:
            not_found()
        bands = image.meta.get('BANDS')
        bdate = list(map(lambda e: e.get('DATE'), bands))

        dataset = gdal.Open(image.file_path, gdal.GA_ReadOnly)
        bcount = dataset.RasterCount

        transform = dataset.GetGeoTransform()
        x = (int)(((float(lng) - transform[0]) / transform[1]))
        y = (int)(((float(lat) - transform[3]) / transform[5]))
        result = []
        for idx in range(bcount):
            band = dataset.GetRasterBand(idx + 1)
            pixel = band.ReadRaster(xoff=x, yoff=y, xsize=1, ysize=1, buf_type=gdal.GDT_Float32)
            import struct
            tuple_of_floats = struct.unpack('f' * 1, pixel)
            result.append({
                'value': tuple_of_floats[0],
                'date': bands[idx].get('DATE'),
                'name': bands[idx].get('ORIGIN_NAME')
            })
        return success(result)

    def image_histogram_calculate(self, wks_id, image_id):
        image = Imagery.query().filter_by(id=image_id).first()
        if not image:
            not_found()
        ds = gdal.Open(image.file_path, gdal.GA_ReadOnly)
        bcount = ds.RasterCount
        result = []
        for i in range(bcount):
            band = ds.GetRasterBand(i + 1)
            bandarr = band.ReadAsArray().astype(float)
            bandarr[bandarr == 0] = np.nan
            max_value = band.GetMaximum()
            min_value = band.GetMinimum()
            hist, bin_edges = np.histogram(bandarr, range=(min_value, max_value), bins=100)
            result.append({
                'hist': list(hist.astype(float)),
                'bin_edges': list(bin_edges.astype(float))
            })
        return success(result)

    def update_cut_line(self, wks_id, prepare_id, payload):
        geojson = payload.get('geojson')
        user_id = payload.get('user_id')
        abs_prepare_path = path_2_abs(payload.get('path'))
        cut_line_json_path = f'{abs_prepare_path}/misc/prepare_cutline_topology.json'
        wks_id = uuid.UUID(wks_id).hex
        prepare_id = uuid.UUID(prepare_id).hex
        dir_path = get_workspace_dir(user_id, wks_id)
        with open(cut_line_json_path, 'w') as json_file:
            json.dump(geojson, json_file)
        image_ids = payload.get('image_ids')
        dir_path_prepare = f'{dir_path}/mosaic_prepare/{prepare_id}'
        input_path = f'{dir_path_prepare}/input.txt'
        input_content = ''
        image_ids = list(map(lambda img_id: uuid.UUID(img_id), image_ids))
        image_ids.reverse()
        for image_id in image_ids:
            input_content += f'\"{dir_path}/{image_id.hex}.tif\"\n'
        with open(input_path, 'w') as cursor:
            cursor.write(input_content)

        df = gpd.read_file(cut_line_json_path, driver='GeoJSON')
        shp_file_path = f'{dir_path_prepare}/prepare/misc/shapefile/cutlines.shp'

        df.to_file(driver='ESRI Shapefile', filename=shp_file_path)

        return success("Success")

    def get_unique_value_pixel(self, user_id, wks_id, img_id, band):
        dir_path = get_workspace_dir(user_id, wks_id)
        image_id = uuid.UUID(img_id)
        image_path = f'{dir_path}/{image_id.hex}.tif'
        with rasterio.open(image_path) as dataset:
            band_array = dataset.read(band)

        return np.unique(band_array)

    def get_color_table(self, user_id, wks_id, img_id, band):
        dir_path = get_workspace_dir(user_id, wks_id)
        image_id = uuid.UUID(img_id)
        image_path = f'{dir_path}/{image_id.hex}.tif'
        datasource = gdal.Open(image_path)
        band1 = datasource.GetRasterBand(band)
        color_table = band1.GetColorTable()
        if not color_table:
            return []
        color_count = color_table.GetCount()
        color_values = []
        for idx in range(color_count):
            r, g, b, _ = color_table.GetColorEntry(idx)
            if r == 0 and g == 0 and b == 0:
                continue
            color_values.append({"color": rgb_2_hex(r, g, b), "minValue": idx})
        del datasource

        return color_values

    def get_range_value_reclassification(self, user_id, wks_id, img_id, band, type, number_class=6):
        dir_path = get_workspace_dir(user_id, wks_id)
        image_id = uuid.UUID(img_id)
        image_path = f'{dir_path}/{image_id.hex}.tif'
        with rasterio.open(image_path) as dataset:
            band_array = dataset.read(band)

        max_value = float(np.nanmax(band_array))
        min_value = float(np.nanmin(band_array[np.nonzero(band_array)]))
        unique_value = np.unique(band_array)

        range_values = []
        if len(unique_value) < number_class:
            number_class = len(unique_value)
            for i in range(number_class):
                range_values.append({
                    'value': i,
                    'minValue': i,
                    'maxValue': i,
                    'percentValue': round((i / max_value) * 100, 6)
                })
            return range_values
        if type == 'quantile':
            with COGReader(image_path) as cog:
                for i in range(number_class):
                    stats = cog.stats(pmax=(i + 1) * 100 / number_class, pmin=i * 100 / number_class)
                    range_values.append({
                        'value': i + 1,
                        'minValue': stats.get(band)['pc'][0],
                        'maxValue': stats.get(band)['pc'][1],
                        'percentValue': round((stats.get(band)['pc'][1] - min_value) / (max_value - min_value) * 100, 6)
                    })
        elif type == 'equal':
            range_value = (max_value - min_value) / number_class
            for i in range(number_class):
                range_values.append({
                    'value': i + 1,
                    'minValue': min_value + range_value * i,
                    'maxValue': min_value + range_value * (i + 1),
                    'percentValue': round(
                        (min_value + (max_value - min_value) / number_class * (i + 1)) / max_value * 100, 6)
                })


        elif type == 'standard':
            array_standard = [0.1, 2.2, 15.8, 50, 84.2, 97.8, 99.9]
            with COGReader(image_path) as cog:
                for i in range(len(array_standard) - 1):
                    stats = cog.stats(pmax=array_standard[i + 1], pmin=array_standard[i])
                    range_values.append({
                        'value': i + 1,
                        'minValue': stats.get(band)['pc'][0],
                        'maxValue': stats.get(band)['pc'][1],
                        'percentValue': round((stats.get(band)['pc'][1] / max_value) * 100, 6)
                    })

        range_values[-1]['maxValue'] = max_value
        range_values[-1]['percentValue'] = 100
        return range_values

    def store_metadata(self, user_id, wks_id, img_id, properties):
        dir_path = get_workspace_dir(user_id, wks_id)
        image_id = uuid.UUID(img_id)
        image_path = f'{dir_path}/{image_id.hex}.tif'
        with rasterio.open(image_path, 'r+') as src_ds:
            # src_ds.update_tags(a=properties)
            src_ds.update_tags(properties=properties)
            properties
            tags = src_ds.tags()

        return dict(tags)

