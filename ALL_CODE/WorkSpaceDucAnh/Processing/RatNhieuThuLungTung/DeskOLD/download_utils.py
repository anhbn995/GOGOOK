import numpy as np
from shapely.geometry import Polygon
from app.workers.utils.tileutils import (getExtentOfPolygon, getGridMatrix, getGridBound, intersect, tileToGoogleTile,
                                         createImage, concurrencyBingTileDownload, concurrencyXYZTileDownload)
from app.utils.imagery import calculate_metadata, _translate
from app.utils.string import get_unique_id
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from app.utils.imagery import stretch
from osgeo import gdal, gdalconst
import uuid
import os, glob
import json, requests
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
api = SentinelAPI('lehai.ha', 'DangKhoa@123')
from google.cloud import storage
from datetime import timedelta
from app.utils.imagery import compute_bound, clip_image
from app.utils.imagery import reproject_image
from app.workers.utils.download_sentinel2_aws import download_jp2s_from_product_id
import urllib.request
import shutil
from config.default import CUBE_HOST, GDAL_BIN, AWS_DATA_FOLDER, SNAP_OPT, XML_FILE_10m, XML_FILE_50m
from zipfile import ZipFile
import subprocess
import rasterio
import time
from datetime import date

def geom_2_polygon(geom):
    points = geom.get('features')[0].get('geometry').get('coordinates')[0]
    polygon = Polygon(points)
    return polygon, points


def get_sen2lv2_from_sen2lv1(sen2lv1_id):
    try:
        arr = sen2lv1_id.split('_')
        # get metadata
        products = api.query(filename=sen2lv1_id + '.SAFE')
        metadata = list(products.items())[0][1]
        # get time period
        d1 = metadata['beginposition']
        d2 = d1 + timedelta(hours=1)
        # get level1 identifier
        level1cpdiidentifier = metadata.get('level1cpdiidentifier')
        if level1cpdiidentifier:
            query = {
                'date': (d1, d2),
                'platformname': 'Sentinel-2',
                'filename': arr[0] + '_MSIL2A_' + arr[2] + '*' + arr[5] + '_*',
                'level1cpdiidentifier': level1cpdiidentifier
            }
        else:
            query = {
                'date': (d1, d2),
                'platformname': 'Sentinel-2',
                'filename': arr[0] + '_MSIL2A_' + arr[2] + '*' + arr[5] + '_*'
            }
        # query level 2

        products = api.query(**query)
        items = list(products.items())
        if len(items) > 0:
            sen2lv2_id = items[0][1]['title']
            return sen2lv2_id
        else:
            raise Exception('Sentinel image lv2a not found, https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a-processing')
    except:
        raise Exception('Sentinel image lv2a not found, https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a-processing')

def download_sentinel2(dir_path, file_id, geom, id, on_processing):
    try:
        tmp_filelist = []
        on_processing(0.1)
        stretch_path = "%s/%s.json" % (dir_path, file_id.hex)

        try:
            sen2lv2_id = get_sen2lv2_from_sen2lv1(id)
            level = 2
        except:
            sen2lv2_id = id
            level = 1

        year, month, day = get_timefrom_title(id, 'sen2')

        temp_path = dir_path + "/tmp/{}".format(file_id.hex)
        reprojected_path = "{}/reprojected_path.tif".format(temp_path)
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        out_path = "%s/%s.tif" % (dir_path, file_id.hex)
    
        temp_id = uuid.UUID(str(get_unique_id()))

        output_file_name = temp_id.hex + "_sen2lv2.tif"
        output_file = temp_path + '/' + output_file_name
        on_processing(0.2)

        filelist = download_jp2s_from_product_id(sen2lv2_id, level=level)
        print(filelist)
        on_processing(0.4)
        gdal_merge = "{}/gdal_merge.py".format(GDAL_BIN)
        from config.default import PYTHON_ENV_PATH
        abs_python_path = PYTHON_ENV_PATH

        #merge files
        merge_command = '{} {} -separate -a_nodata 0 -o {} {} {} {} {}'.format(abs_python_path, gdal_merge,
            output_file, *filelist)
        os.system(merge_command)
        on_processing(0.6)
        
        reproject_image(output_file, reprojected_path)

        # gen COG file and clip
        if geom:
            computed_geom = geom
            clip_image(geom, reprojected_path, str(file_id), dir_path)
        else:
            _translate(reprojected_path, out_path)
            computed_geom = compute_bound(out_path)

        on_processing(0.8)
        stretch(out_path, stretch_path, mode=1)
        on_processing(0.95)

        os.remove(reprojected_path)
        for file in tmp_filelist:
            os.remove(file)
        shutil.rmtree(temp_path)
        return date(year, month, day), computed_geom
    except Exception as e:
        print('Exception', e)
        try:
            if not image_path and os.path.exists(output_file):
                os.remove(output_file)
            if reprojected_path and os.path.exists(reprojected_path):
                os.remove(reprojected_path)
            shutil.rmtree(temp_path)
        except:
            pass
        raise e

def download_sentinel1_aws(dir_path, file_id, geom, id, temp_dir,product, on_processing):
    try:
        year, month, day = get_timefrom_title(id, 'sen1')
        link = 's3://sentinel-s1-l1c/GRD/{}/{}/{}/IW/DV/{}'.format(year,month,day,id)
        data_dir = '{}/{}.zip'.format(AWS_DATA_FOLDER, id)
        product_folder = '{}/{}'.format(AWS_DATA_FOLDER, product)
        if not os.path.exists(product_folder):
            os.makedirs(product_folder)
            
        aws_tif_path = '{}/{}.tif'.format(product_folder, id)
        
        out_path = "{}/{}.tif" .format(dir_path, file_id.hex)
        stretch_path = "{}/{}.json" .format(dir_path, file_id.hex)
        
        cache_id = uuid.uuid4()
        cache_dir = '{}/{}'.format(temp_dir, cache_id.hex)
        data_cache_dir = '{}/{}.SAFE'.format(cache_dir, id)
        os.makedirs(data_cache_dir)

        if os.path.exists(data_dir):
            print('File zip already exist')
            shutil.unpack_archive(data_dir, cache_dir, 'zip')
            time.sleep(10)
            pass
        else:
            print('Started downloading ...')

            command = 'aws s3 cp {} {} --recursive --request-payer'.format(link, data_cache_dir)
            os.system(command)

            print('Starting snap')
            # os.rename(data_cache_dir, data_cache_dir+'.SAFE')
            shutil.make_archive('{}/{}'.format(AWS_DATA_FOLDER, id), 'zip', cache_dir)

        if os.path.exists(aws_tif_path):
            pass
        else:
            if product == 'sentinel_1_grd_50m_beta0':
                xml_file = XML_FILE_50m
                param = 5

            else:
                xml_file = XML_FILE_10m
                param = 1

            cache_tif_path = '{}/{}.tif'.format(cache_dir, cache_id.hex)
            # cache_tif_path = '/media/boom/data1/geoai_storage/data/21/c69f2164de33480fa1a4c81e6f723501/tmp/2724333b8482456eab7449a901b956b4/2724333b8482456eab7449a901b956b4.tif'
            snap_command = "{} {} -Pinputfile={} -Poutputfile={}".format( SNAP_OPT, xml_file, data_dir, cache_tif_path)
            os.system(snap_command)
            print('Snapping successful')
            # rename some files
            vv_name, vh_name = rename(id)
            print('Rename successful')

            anno_path = '{}/annotation'.format(data_cache_dir)
            for file in glob.glob('{}/*.xml'.format(anno_path)):
                file_name = file.split('/')[-1]
                format_str = file_name.split('.')[-1]
                if 'vv' in file_name:
                    os.rename(file,'{}/{}.{}'.format(anno_path,vv_name, format_str))
                if 'vh' in file_name:
                    os.rename(file,'{}/{}.{}'.format(anno_path,vh_name, format_str))

            
            measure_path = '{}/measurement'.format(data_cache_dir)
            for file in glob.glob('{}/*f'.format(measure_path)):
                file_name = file.split('/')[-1]
                format_str = file_name.split('.')[-1]
                if format_str in ['tif','tiff']:
                    if 'vv' in file_name:
                        os.rename(file,'{}/{}.{}'.format(measure_path,vv_name, format_str))
                    if 'vh' in file_name:
                        os.rename(file,'{}/{}.{}'.format(measure_path,vh_name, format_str))
                else:
                    pass

            # #copy project and metadata
            print('Compute GCP ...')
            # Open the file:
            source_ds= gdal.Open(data_cache_dir , gdalconst.GA_ReadOnly)
            metadata=source_ds.GetMetadata()

            gcp = source_ds.GetGCPs()
            gcpproj = source_ds.GetGCPProjection()
            
            ds = gdal.Open( cache_tif_path, gdalconst.GA_Update )

            # resolution from 10m to x*10m
            newgcp=[gdal.GCP(tmp.GCPX, tmp.GCPY, tmp.GCPZ, tmp.GCPPixel//param, tmp.GCPLine//param) for tmp in gcp]

            ds.SetGCPs( newgcp, gcpproj )
            ds.SetMetadata(metadata)

            ds=None
            source_ds=None
            
            kwargs = {'format': 'GTiff', 'dstSRS': 'EPSG:4326', 'dstNodata':'0'}
            gdal.Warp(aws_tif_path, cache_tif_path,**kwargs)

        shutil.rmtree(cache_dir)

        if geom:
            print('Clipping image ...')
            computed_geom = geom
            clip_image(geom, aws_tif_path, str(file_id), dir_path)
        else:
            _translate(aws_tif_path, out_path)
            computed_geom = compute_bound(out_path)
        stretch(out_path, stretch_path, mode=1)
        
        return computed_geom, date(year, month, day)
    except:
        raise Exception('Sentinel1 image not found')

def get_timefrom_title(title, type):
    list_str = title.split('_')
    if type == 'sen1':
        sub_str = list_str[4]
    elif type == 'sen2':
        sub_str = list_str[2]

    year = int(sub_str[:4])
    month = int(sub_str[4:6])
    day = int(sub_str[6:8])
    return  year,month,day

def rename(src_name):
    str1=src_name.lower()
    str2=str1.replace('_','-')
    sub_str = str2[17:62]
    vh_subfix = 's1a-iw-grd-vh-{}-002'.format(sub_str)
    vv_subfix = 's1a-iw-grd-vv-{}-001'.format(sub_str)
    return vv_subfix, vh_subfix

def download_planet(dir_path, list_result, on_processing):
    cache_id = uuid.uuid4()
    cache_dir = '{}/{}'.format(dir_path, cache_id.hex)
    os.makedirs(cache_dir)

    res = []
    for result in list_result:
        name = result.get('name').split("/")[-1]

        url = result.get('location')
        sub_str_list = ["Analytic_clip.tif", "analytic_clip.tif"]
        for sub_str in sub_str_list:
            if sub_str in name:
                file_id = uuid.uuid4()
                file_cache_path = '{}/{}'.format(cache_dir, name)
                urllib.request.urlretrieve(url, file_cache_path)

                preproject_path = '{}/{}_tmp.tif'.format(cache_dir, file_id.hex)
                reproject_image(file_cache_path, preproject_path)

                final_file_path = '{}/{}.tif'.format(dir_path, file_id.hex)
                stretch_path = '{}/{}.json'.format(dir_path, file_id.hex)
                _translate(preproject_path, final_file_path)
                stretch(final_file_path, stretch_path, mode=1)
                res.append({
                    'file_path': final_file_path,
                    'file_id': str(file_id)
                })
    shutil.rmtree(cache_dir)
        
    return res

def download_modis(dir_path, url, on_processing):
    cache_id = uuid.uuid4()
    cache_dir = '{}/{}'.format(dir_path, cache_id.hex)
    os.makedirs(cache_dir)
    zip_cache_path = '{}/{}.tar.gz'.format(cache_dir, cache_id.hex)
    urllib.request.urlretrieve(url, zip_cache_path)
    
    import tarfile
    tar = tarfile.open(zip_cache_path)
    tar.extractall(cache_dir)

    file_list = glob.glob("{}/*1.tif".format(cache_dir))
    list_str_prefix = []
    for file in file_list:
        name_file = file.split("/")[-1]
        str_prefix = name_file[:-5]
        list_str_prefix.append(str_prefix)

    res = []
    for str_prefix in list_str_prefix:
        sub_list_file = glob.glob("{}/{}*".format(cache_dir, str_prefix))
        sort_list = list(np.sort(sub_list_file))
        file_id = uuid.uuid4()
        str_file_id = file_id.hex

        file_cache_path = '{}/{}.tif'.format(cache_dir, str_file_id)
        abs_vrt_path = '{}/{}.vrt'.format(cache_dir, str_file_id)

        gdal.BuildVRT(abs_vrt_path , sort_list, options=gdal.BuildVRTOptions(separate=True))
        gdal.Translate(file_cache_path, abs_vrt_path, format="GTiff", creationOptions=['COMPRESS=LZW'])

        preproject_path = '{}/{}_tmp.tif'.format(cache_dir, str_file_id)
        reproject_image(file_cache_path, preproject_path)

        final_file_path = '{}/{}.tif'.format(dir_path, file_id.hex)
        stretch_path = '{}/{}.json'.format(dir_path, file_id.hex)

        on_processing(0.8)
        _translate(preproject_path, final_file_path)
        stretch(final_file_path, stretch_path, mode=1, min=40, max=60)
        on_processing(0.95)
        res.append({
            'file_path': final_file_path,
            'file_id': str(file_id)
        })

    shutil.rmtree(cache_dir)
    return res

def download_landsat8(dir_path, _id, geom, on_processing):
    cache_id = uuid.uuid4()
    str_id = cache_id.hex
    cache_dir = '{}/{}'.format(dir_path, str_id)
    os.makedirs(cache_dir)

    host = "http://landsat-pds.s3.amazonaws.com/c1/L8/"
    sub_str = _id.split("_")
    WRS_path = sub_str[2][:3]
    WRS_row = sub_str[2][3:]
    for i in [2, 3, 4, 5, 8]:
        name = "{}_B{}.TIF".format(_id, i)
        url = host+ WRS_path+ "/"+ WRS_row + "/"+ _id+ "/"+ name
        zip_cache_path = '{}/{}.tif'.format(cache_dir, name)
        urllib.request.urlretrieve(url, zip_cache_path)

    file_list = glob.glob("{}/*.tif".format(cache_dir))
    sort_list = list(np.sort(file_list))

    file_cache_path = '{}/{}.tif'.format(cache_dir, str_id)
    abs_vrt_path = '{}/{}.vrt'.format(cache_dir, str_id)
    res = []
    on_processing(0.3)

    panchro_path = sort_list[-1]
    list_bands = sort_list[:-1]

    list_call= ["{}/gdal_pansharpen.py".format(GDAL_BIN), "-of", "vrt"]
    list_call.append(panchro_path)
    list_call.extend(list_bands)
    list_call.append(abs_vrt_path)

    subprocess.call(list_call)
    gdal.Translate(file_cache_path, abs_vrt_path, format="GTiff", noData=0, creationOptions=['COMPRESS=LZW'])

    on_processing(0.6)
    preproject_path = '{}/{}_tmp.tif'.format(cache_dir, str_id)
    reproject_image(file_cache_path, preproject_path)
    final_file_path = '{}/{}.tif'.format(dir_path, str_id)

    on_processing(0.7)
    # gen COG file and clip
    if geom:
        clip_image(geom, preproject_path, str(cache_id), dir_path)
    else:
        _translate(preproject_path, final_file_path)

    on_processing(0.8)
    stretch_path = '{}/{}.json'.format(dir_path, str_id)
    stretch(final_file_path, stretch_path, mode=1, min=2, max=98)
    on_processing(0.95)

    res.append({
        'file_path': final_file_path,
        'file_id': str(cache_id)
    })

    shutil.rmtree(cache_dir)
    return res

def download_skywatch(dir_path, url, geom, on_processing):
    cache_id = uuid.uuid4()
    str_id = cache_id.hex
    cache_dir = '{}/{}'.format(dir_path, str_id)
    os.makedirs(cache_dir)
    file_cache_path = '{}/{}.tif'.format(cache_dir, str_id)

    on_processing(0.4)
    urllib.request.urlretrieve(url, file_cache_path)

    on_processing(0.6)
    preproject_path = '{}/{}_tmp.tif'.format(cache_dir, str_id)
    reproject_image(file_cache_path, preproject_path)
    final_file_path = '{}/{}.tif'.format(dir_path, str_id)
    res = []
    on_processing(0.7)
    # gen COG file and clip
    if geom:
        clip_image(geom, preproject_path, str(cache_id), dir_path)
    else:
        _translate(preproject_path, final_file_path)

    on_processing(0.8)
    stretch_path = '{}/{}.json'.format(dir_path, str_id)
    stretch(final_file_path, stretch_path, mode=1)
    on_processing(0.95)

    res.append({
        'file_path': final_file_path,
        'file_id': str(cache_id)
    })

    shutil.rmtree(cache_dir)
    return res

def download_datasource(dir_path, path, geom, on_processing):
    on_processing(0.2)
    file_id = uuid.uuid4()
    final_file_path = '{}/{}.tif'.format(dir_path, file_id.hex)
    if geom:
        clip_image(geom, path, str(file_id), dir_path)
    else:
        _translate(path, final_file_path)

    res = []
    on_processing(0.7)
    stretch_path = '{}/{}.json'.format(dir_path, file_id.hex)
    stretch(final_file_path, stretch_path, mode=2)
    on_processing(0.95)

    res.append({
        'file_path': final_file_path,
        'file_id': str(file_id)
    })
    
    return res

def download_xyz(url, dir_path, file_id, geom, zoom_level, projector, crs, on_processing):
    temp_id = uuid.UUID(str(get_unique_id()))
    temp_path = dir_path + "/tmp"

    output_file_name = temp_id.hex + "_bingmap.tif"
    output_file = temp_path + '/' + output_file_name
    log_file = os.path.join(temp_path, temp_id.hex + "_downloaded.txt")
    
    try:
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        polygon, points = geom_2_polygon(geom)
        grid_matrix, width, height = compute_grid_matrix(points, zoom_level)
        google_grid_matrix, grid_index_list, new_bound = prepare_data_download_bing(grid_matrix, width, height, polygon,
                                                                                    projector)

        dataset_location = createImage(256 * width, 256 * height, 4, new_bound, output_file, crs=crs)
        concurrencyXYZTileDownload(url, grid_index_list, google_grid_matrix, dataset_location, log_file, on_processing)
        clipped_image_data = clip_image(geom, output_file, str(file_id), dir_path)

        os.remove(output_file)
        os.remove(log_file)
        return output_file
    except Exception as e:
        if os.path.exists(output_file):
            os.remove(output_file)
        if os.path.exists(log_file):
            os.remove(log_file)
        raise e

def download_bing(dir_path, file_id, geom, zoom_level, projector, crs, on_processing):
    temp_id = uuid.UUID(str(get_unique_id()))
    temp_path = dir_path + "/tmp"

    output_file_name = temp_id.hex + "_bingmap.tif"
    output_file = temp_path + '/' + output_file_name
    log_file = os.path.join(temp_path, temp_id.hex + "_downloaded.txt")
    
    try:
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        polygon, points = geom_2_polygon(geom)
        grid_matrix, width, height = compute_grid_matrix(points, zoom_level)
        google_grid_matrix, grid_index_list, new_bound = prepare_data_download_bing(grid_matrix, width, height, polygon,
                                                                                    projector)

        dataset_location = createImage(256 * width, 256 * height, 4, new_bound, output_file, crs=crs)
        concurrencyBingTileDownload(grid_index_list, google_grid_matrix, dataset_location, log_file, on_processing)
        clipped_image_data = clip_image(geom, output_file, str(file_id), dir_path)

        os.remove(output_file)
        os.remove(log_file)
        return output_file
    except Exception as e:
        if os.path.exists(output_file):
            os.remove(output_file)
        if os.path.exists(log_file):
            os.remove(log_file)
        raise e

def prepare_data_download_bing(grid_matrix, width, height, polygon, projector):
    bound_grid_matrix = np.reshape(np.apply_along_axis(getGridBound, 1, grid_matrix.reshape(width * height, 3)),
                                   (height, width, 4))
    intersect_grid_matrix = np.reshape(
        np.apply_along_axis(intersect, 1, bound_grid_matrix.reshape(width * height, 4), polygon),
        (height, width))
    google_grid_matrix = np.reshape(
        np.apply_along_axis(tileToGoogleTile, 1, grid_matrix.reshape(width * height, 3)),
        (height, width, 3)).astype(np.uint32)
    grid_index_list = np.array(np.where(intersect_grid_matrix)).transpose()
    new_bound = bound_grid_2_bound(bound_grid_matrix, projector)
    return google_grid_matrix, grid_index_list, new_bound

def compute_grid_matrix(points, zoom_level):
    extent = getExtentOfPolygon(points)
    offset = getGridMatrix(extent, zoomLevel=zoom_level)
    grid_matrix = np.zeros((offset[3] - offset[1] + 1, offset[2] - offset[0] + 1, 3), dtype=np.uint32)
    grid_matrix, width, height = apply_offset_to_grid_matrix(grid_matrix, offset, zoom_level)
    return grid_matrix, width, height

def apply_offset_to_grid_matrix(grid_matrix, offset, zoom_level):
    height, width, _ = grid_matrix.shape
    for i in range(height):
        for j in range(width):
            grid_matrix[i, j] = (offset[0] + j, offset[3] - i, zoom_level)
    return grid_matrix, width, height

def bound_grid_2_bound(bound_grid_matrix, proj):
    west = bound_grid_matrix[0, 0][0]
    north = bound_grid_matrix[0, 0][3]
    east = bound_grid_matrix[-1, -1][2]
    south = bound_grid_matrix[-1, -1][1]
    west, north = proj(west, north)
    east, south = proj(east, south)
    return (west, south, east, north)

def get_image_meta(geom, dir_path, file_id):
    meta = calculate_metadata(dir_path + '/' + file_id.hex + '.tif')
    meta["img_path"] = dir_path
    meta["file_id"] = str(file_id)
    meta["geometry"] = geom
    return meta

def translate(src_path, dst_path, profile="deflate", profile_options={}, **options):
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
        in_memory=False,
        quiet=True,
        **options,
    )
    return True