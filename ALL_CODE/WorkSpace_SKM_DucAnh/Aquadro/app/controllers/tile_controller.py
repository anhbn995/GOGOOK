import io
import os

import numpy as np
from flask import send_file
from flask import request
from rio_tiler.utils import render
# from rio_tiler_pds.sentinel.aws import (
#     S2COGReader,  # COG
#     S1L1CReader
# )
from lib.image_reader.readers.aws.sentinel2 import S2COGReader
from app.services.image_readers import S1L1CReader
from rio_tiler.io import COGReader
from app.api.cache import cache
from app.utils.indices_helper import *
from app.utils.color_helper import hex_2_rgb
from app.utils.tile_helper import crop_tile

from app.utils.response import success
from config.app import PLANET_FOLDER

def _stretch(tile, stretch=None):
    bcount = len(tile)
    output = np.zeros(np.shape(tile))
    for idx in range(bcount):
        band = np.array(tile[idx])
        nodatamask = (band == np.nan)
        if stretch:
            _min = (stretch[idx][0])
            _max = (stretch[idx][1])
        else:
            _min = np.percentile(band, 2)
            _max = np.percentile(band, 98)
        band = np.interp(band, (_min, _max), (1, 255)).astype(np.uint8)
        band[nodatamask] = 0
        output[idx] = band
    output = output.astype(np.uint8)
    return output


def sentinel2_statistic(img_id):
    with S2COGReader(img_id) as sentinel:
        return sentinel.statistics(bands=("B02", "B03", "B04", "B08"))


@cache.memoize()
def get_stretch(sentinel):
    stats = sentinel.statistics(bands=("B04", "B03", "B02"))
    stretch = [[stats['B04'].percentile_2, stats['B04'].percentile_98],
               [stats['B03'].percentile_2, stats['B03'].percentile_98],
               [stats['B02'].percentile_2, stats['B02'].percentile_98]]
    return stretch


def _sentinel2_generate_tile(img_id, tile_z, tile_x, tile_y):
    with S2COGReader(img_id) as sentinel:
        data, mask = sentinel.tile(tile_x, tile_y, tile_z, bands=("B04", "B03", "B02"))
        stretch = get_stretch(sentinel)
        data = _stretch(data, stretch)
        tile = render(data.astype(np.uint8), mask=mask, img_format='png')
    return send_file(io.BytesIO(tile), mimetype='image/png', download_name='{}.png'.format(y))


def sentinel_serve_tile(img_id, z, x, y):
    payload = request.args
    fields = payload.get('fields')
    aoi = payload.get('aoi')
    index = payload.get('index', None)
    if index not in INDICES:
        bands = ("B04", "B03", "B02")
        expression = None
    else:
        bands = None
        expression = index_expression(index)

    tile_z = int(z)
    tile_x = int(x)
    tile_y = int(y)
    if tile_z < 12:
        return success("Cannot load tile with zoom level less than 12")
    if bands is not None or index_source(index) == SENTINEL2:
        with S2COGReader(img_id) as sen2:
            tile_data = sen2.tile(tile_x, tile_y, tile_z, bands=bands, expression=expression, tilesize=256,
                                  force_binary_mask=False)
            mask = tile_data.mask
    elif index_source(index) == SENTINEL1:
        os.environ['AWS_REQUEST_PAYER'] = 'requester'
        with S1L1CReader(img_id) as sen1:
            tile_data = sen1.tile(tile_x, tile_y, tile_z, expression=expression, tilesize=256,
                                  force_binary_mask=False)
            mask = tile_data.mask

    if fields or aoi:
        cropped_tile = crop_tile(tile_data, fields=fields, aoi=aoi)
        mask = (cropped_tile[0] != float(0)) * 1 * 255

    if expression:
        tile, mask = tile_pseudocolor(tile_data.data[0], mask, index)
    else:
        stretch = get_stretch(sen2)
        tile = _stretch(tile_data.data, stretch)

    tile = render(tile.astype(np.uint8), mask=mask, img_format='png')
    return send_file(io.BytesIO(tile), mimetype='image/png', download_name='{}.png'.format(y))


def sentinel2_cloud_tile(img_id, z, x, y):
    def cloud_mask(cmask):
        return np.logical_or(cmask[0] == 8, cmask[0] == 9) * 1 * 255

    payload = request.args
    fields = payload.get('fields', None)
    aoi = payload.get('aoi', None)

    tile_z = int(z)
    tile_x = int(x)
    tile_y = int(y)
    if tile_z < 12:
        return success("Cannot load tile with zoom level less than 12")

    with S2COGReader(img_id) as sen2:
        scl_path = sen2._get_band_url("B01").replace("B01", "SCL")

    with COGReader(scl_path) as cog:
        tile_data = cog.tile(tile_x, tile_y, tile_z, tilesize=256, force_binary_mask=False)

    if fields or aoi:
        cropped_tile = crop_tile(tile_data, fields=fields, aoi=aoi)
        mask = cloud_mask(cropped_tile)
    else:
        mask = cloud_mask(tile_data.data)

    tile = np.array([mask, mask, mask])
    tile = render(tile.astype(np.uint8), mask=mask, img_format='png')
    return send_file(io.BytesIO(tile), mimetype='image/png', download_name='{}.png'.format(y))


def tile_pseudocolor(tile, mask, index):
    r_band = np.zeros(mask.shape, dtype=int)
    g_band = np.zeros(mask.shape, dtype=int)
    b_band = np.zeros(mask.shape, dtype=int)

    color_list = index_colors(index)
    for index, color in enumerate(color_list):
        if index == (len(color_list) - 1):
            logical_mask = tile > float(color[0])
        else:
            next_color = color_list[index + 1]
            logical_mask = np.logical_and(tile > float(color[0]), tile <= float(next_color[0]))
        palette = hex_2_rgb(color[1])
        r_band[logical_mask] = palette[0]
        g_band[logical_mask] = palette[1]
        b_band[logical_mask] = palette[2]

    mask[tile <= color_list[0][0]] = 0
    return np.stack((r_band, g_band, b_band)).astype(np.uint8), mask


def planet_tile(image_id, z, x, y):
    payload = request.args
    field_id = payload.get('fields').split('_')[1].replace('-', '')
    index = payload.get('index', 'NDVI')
    expression = index_expression(index, PLANET)

    tile_z = int(z)
    tile_x = int(x)
    tile_y = int(y)
    if tile_z < 12:
        return success("Cannot load tile with zoom level less than 12")

    with COGReader(f"{PLANET_FOLDER}/{field_id}/{image_id}_3B_AnalyticMS_SR_8b_clip.tif") as image:
        data, mask = image.tile(tile_x, tile_y, tile_z, expression=expression, tilesize=256, force_binary_mask=False)

    tile, mask = tile_pseudocolor(data[0], mask, index)
    tile = render(tile.astype(np.uint8), mask=mask, img_format='png')
    return send_file(io.BytesIO(tile), mimetype='image/png', download_name='{}.png'.format(y))


def planet_cloud_tile(image_id, z, x, y):
    payload = request.args
    field_id = payload.get('fields').split('_')[1].replace('-', '')
    tile_z = int(z)
    tile_x = int(x)
    tile_y = int(y)
    if tile_z < 12:
        return success("Cannot load tile with zoom level less than 12")

    with COGReader(f"{PLANET_FOLDER}/{field_id}/{image_id}_3B_udm2_clip.tif") as image:
        data, mask = image.tile(tile_x, tile_y, tile_z, indexes=6, tilesize=256, force_binary_mask=False)

    mask = data[0]*255
    data = np.array([data[0]*255, data[0]*255, data[0]*255])
    tile = render(data.astype(np.uint8), mask=mask, img_format='png')
    return send_file(io.BytesIO(tile), mimetype='image/png', download_name='{}.png'.format(y))

