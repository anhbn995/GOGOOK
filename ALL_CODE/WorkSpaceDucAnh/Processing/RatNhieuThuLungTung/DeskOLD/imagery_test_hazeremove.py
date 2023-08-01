import os
import json
import uuid

import numpy as np
import rasterio
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

from app.utils.haze_removal import haze_removal_tools


def _translate(src_path, dst_path, profile="deflate", profile_options={}, **options):
    output_profile = cog_profiles.get(profile)
    output_profile.update(dict(BIGTIFF="IF_SAFER"))
    output_profile.update(profile_options)
    config = dict(
        GDAL_NUM_THREADS="ALL_CPUS",
        GDAL_TIFF_INTERNAL_MASK=True,
        GDAL_TIFF_OVR_BLOCKSIZE="128",
    )
    cog_translate(
        src_path,
        dst_path,
        output_profile,
        config=config,
        quiet=True,
        add_mask=False,
        **options,
    )
    return True


def haze_removal_image_tools(image_id, result_id, out_dir, on_processing=None):
    haze_name = 'pre_translate_{}'.format(result_id.hex)
    haze_result_path = '{}/{}.tif'.format(out_dir, haze_name)

    haze_removal_tools(image_id.hex, haze_name, out_dir)

    out_path = '{}/{}.tif'.format(out_dir, result_id.hex)
    result_stretch_path = '{}/{}.json'.format(out_dir, result_id.hex)
    _translate(haze_result_path, out_path)

    stretch(haze_result_path, result_stretch_path, mode=1)

    os.remove(haze_result_path)


def stretch_without_json(input, mode=1, min=2, max=98, nodata=0):
    mode = int(mode)
    min = int(min)
    max = int(max)
    from osgeo import gdal_array
    raster = gdal_array.LoadFile(input)
    if len(np.shape(raster)) == 2:
        raster = [raster]
    bcount, _, _ = np.shape(raster)
    strecth_arr = []
    for i in range(bcount):
        band = raster[i]
        nodatamask = (band == nodata)
        band1 = band.astype(float)
        band1[nodatamask] = np.nan
        nodatamask = (band == -9999)
        band1[nodatamask] = np.nan
        if mode == 1:  # cumulative
            p2 = (np.nanpercentile(band1, min))
            p98 = (np.nanpercentile(band1, max))
        else:  # standard deviation
            mean = np.nanmean(band1)
            std = np.nanstd(band1)
            p2 = mean - std * 2
            p98 = mean + std * 2
        strecth_arr.append({'p2': p2, 'p98': p98})
    return strecth_arr


def stretch(input, output, mode=1, min=2, max=98):
    nodata = 0
    with rasterio.open(input) as ds:
        nodata = ds.nodata
    strecth_arr = stretch_without_json(input=input, mode=mode, min=min, max=max, nodata=nodata)
    data = {
        'stretches': strecth_arr
    }
    print(output)
    with open(output, 'w') as outfile:
        json.dump(data, outfile)


if __name__ == '__main__':
    image_id = uuid.UUID('1ebed191-d422-4b7a-8a36-3a5e0454419b')
    out_dir = r'/home/thang/Downloads'
    result_id = uuid.UUID('4120e96b-bfd3-419b-9ac8-2a25ede9966a')
    haze_removal_image_tools(image_id, result_id, out_dir)
