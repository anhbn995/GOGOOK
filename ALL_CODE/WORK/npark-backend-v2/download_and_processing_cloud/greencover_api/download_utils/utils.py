import os
import json
import uuid
from osgeo import gdal, osr, ogr
import fiona
import rasterio
import datetime
import xmltodict
import geopandas as gpd


from os import environ
from datetime import timedelta
from skyamqp import AMQP_Client
from osgeo.gdal import GA_ReadOnly
from rio_tiler.io import COGReader
from sentinelsat import SentinelAPI
from shapely.geometry import Polygon
from rio_cogeo.cogeo import cog_translate
from shapely.geometry import shape, mapping
from rio_cogeo.profiles import cog_profiles

SENTINEL_API_USER="lehai.ha"
SENTINEL_API_PASSWORD="DangKhoa@123"
SENTINEL_API_URL="https://apihub.copernicus.eu/apihub/"

CONDA_PREFIX = (os.environ.get('CONDA_PREFIX') or '/opt/conda')
GDAL_BIN = CONDA_PREFIX + "/bin"
NAS_TEMP_FOLDER = "/home/geoai/geoai_data_test2/nas_temp"

os.environ['AMQP_HOST']='192.168.4.100'
os.environ['AMQP_PORT']='5672'
os.environ['AMQP_VHOST']='/eof'
os.environ['AMQP_USERNAME']='eof_rq_worker'
os.environ['AMQP_PASSWORD']='123'
connection = AMQP_Client(
  host=os.environ.get('AMQP_HOST'),
  port=os.environ.get('AMQP_PORT'),
  virtual_host=os.environ.get('AMQP_VHOST'),
  username=os.environ.get('AMQP_USERNAME'),
  password=os.environ.get('AMQP_PASSWORD'),
  heartbeat=5
)

def convert_l1c_to_l2a(sen2lv1_ids):
    api = SentinelAPI(SENTINEL_API_USER, SENTINEL_API_PASSWORD, SENTINEL_API_URL)
    response = []
    for sen2lv1_id in sen2lv1_ids:
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
                response.append({
                    'L1C': sen2lv1_id,
                    'L2A': items[0][1]['title']
                })
            else:
                response.append({
                    'L1C': sen2lv1_id,
                    'L2A': None
                })
        except:
            response.append({
                'L1C': sen2lv1_id,
                'L2A': None
            })
    return response

def last_day_of_month(any_day):
    # this will never fail
    # get close to the end of the month for any day, and add 4 days 'over'
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)
    # subtract the number of remaining 'overage' days to get last day of current month, or said programattically said, the previous day of the first of next month
    return next_month - datetime.timedelta(days=next_month.day)

def get_raw_image(resp, is_private=False):
    raw_image = {'path': resp}
    if 'save_path' in resp:
        raw_image = {'path': resp['save_path']}
    raw_image['name'] = os.path.basename(raw_image['path'])
    if 'name' in resp:
        raw_image['name'] = resp['name']
    if 'id' in resp:
        raw_image['id'] = resp['id']
    raw_image['is_private'] = is_private

    if os.path.isfile(raw_image['path']):
        extension = os.path.basename(raw_image['path']).split(".")[-1]
        raw_image['type'] = extension
        raw_image['size'] = os.stat(raw_image['path']).st_size

    else:
        raw_image['type'] = 'folder'

    return raw_image

def process_sentinel2_pci(attempts, temp_dir, product, code, raw_path, pre_out_path, has_hazerem=False):
    while attempts < 4:
        try:
            ids = uuid.uuid4().hex
            pix_file = f"{temp_dir}/result_{code}_{ids}.pix"
            meta_file = 'MTD_MSI{}.xml'.format(product)
            gxl_rpcClient = connection.create_RPC_Client('gxl-python')
            resp = gxl_rpcClient.send('fimport', {
                'fili': '{}/{}?r=%3ABand+Resolution%3A{}'.format(raw_path, meta_file, code),
                'filo': pix_file
            })
            print('{}/{}?r=%3ABand+Resolution%3A{}'.format(raw_path, meta_file, code))
            if not resp['success']:
                print(Exception(resp['message']))
                raise Exception('GXL error: ' + resp['message'])

            if has_hazerem:
                if code == '10M' and product == 'L1C':
                    resp = gxl_rpcClient.send('hazerem', {
                        'fili': pix_file,
                        'filo': temp_dir + '/hazefree.pix',
                        'hazecov': [50],
                        'clthresh': [18, 22, 1]
                    })
                    if not resp['success']:
                        print(Exception(resp['message']))
                        raise Exception(resp['message'])
                    pix_file = temp_dir + '/hazefree.pix'
                else:
                    print(Exception('Haze remove is not supported!!'))
                    raise Exception('Haze remove is not supported!!')

            resp = gxl_rpcClient.send('datamerge', {
                'mfile': pix_file,
                'filo': pre_out_path,
                'dbic': get_bands(code, product),
                'ftype': 'TIF'
            })
            if not resp['success']:
                print(Exception(resp['message']))
                raise Exception('GXL error: ' + resp['message'])

            break
        except Exception as e:
            print(e)
            attempts += 1
    return attempts

def make_nas_temp_folder_in_root_data_folder():
    temp_folder = uuid.uuid4().hex
    path = '{}/{}'.format(NAS_TEMP_FOLDER, temp_folder)
    os.makedirs(path)
    return path

def clip_image(aoi, image_path, out_id, out_dir, temp_dir, extra=''):    
    temp_dir = os.path.abspath(temp_dir)
    aoi_path = '{}/{}.geojson'.format(temp_dir, uuid.UUID(out_id).hex)
    out_path = '{}/{}{}.tif'.format(out_dir, uuid.UUID(out_id).hex, extra)
    cutline_path = '{}/{}.shp'.format(temp_dir, uuid.UUID(out_id).hex)
    pretranslate_out_path = '{}/pre_translate_{}.tif'.format(temp_dir, uuid.UUID(out_id).hex)
    print(aoi)
    with open(aoi_path, "w") as editor:
        editor.write(json.dumps(aoi))
    try:
        with fiona.open(aoi_path):
            pass
    except Exception as e:
        print(e)
        with open(aoi_path, "w") as editor:
            editor.write(json.dumps({
                'type': 'FeatureCollection',
                'features': [{
                    'type': 'Feature',
                    'geometry': aoi
                }]
            }))
    data = gdal.Open(image_path, GA_ReadOnly)
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
            print(aoi_polygon)
            bounds.append(aoi_polygon.intersection(origin_extend))
    
    # Define a polygon feature geometry with one attribute
    features = []
    for bound in bounds:
        if not bound.is_empty:
            features.append({
                'type': 'Feature',
                'geometry': mapping(bound)
            })

    feature_collection = {
        'type': 'FeatureCollection',
        'features': features
    }

    if len(features) == 0:
        raise Exception('Not intersect')
        return None

    # Write a new Shapefile
    with open(cutline_path, "w") as editor:
        editor.write(json.dumps(feature_collection))

    with rasterio.open(image_path) as ds:
        nodata = ds.nodata or 0
    gdal.Warp(
        pretranslate_out_path,
        image_path,
        cutlineDSName=cutline_path,
        cropToCutline=True,
        dstNodata=nodata,
        dstSRS='EPSG:4326'
    )
    if os.path.exists(pretranslate_out_path):
        _translate(pretranslate_out_path, out_path)
        print('path', out_path)
        return {
            'geom': aoi,
            'id': out_id,
            'path': out_path
        }
    else:
        return False

def get_bands_for_gdal(code, product):
    if code == '10M':
        return ['R10m/B02', 'R10m/B03', 'R10m/B04', 'R10m/B08']
    if code == '20M':
        if product == 'L1C':
            return ['R20m/B05', 'R20m/B06', 'R20m/B07', 'R20m/B8A', 'R20m/B11', 'R20m/B12']
        else:
            return ['R20m/AOT', 'R20m/B02', 'R20m/B03', 'R20m/B04', 'R20m/B05', 'R20m/B06', 
            'R20m/B07', 'R20m/B8A', 'R20m/B11', 'R20m/B12', 'R20m/SCL', 'R20m/WVP']
    if code == '60M':
        if product == 'L1C':
            return ['R60m/B01', 'R60m/B09', 'R60m/B10']
        else:
            return ['R60m/B01', 'R60m/B02', 'R60m/B03', 'R60m/B04', 'R60m/B05', 'R60m/B06', 
            'R60m/B07', 'R60m/B8A', 'R60m/B09', 'R60m/B11', 'R60m/B12', 'R60m/AOT', 'R60m/SCL', 'R60m/WVP']
    raise Exception(f"No support spatial resolution {code} with {product}")

def gdal_merge_file(path, output_file, bands, level):
    meta_path = "{}/MTD_MSIL1C.xml".format(path)
    xml_key = "n1:Level-1C_User_Product"
    if level == 2:
        meta_path = "{}/MTD_MSIL2A.xml".format(path)
        xml_key = "n1:Level-2A_User_Product"
    else:
        bands = list(map(lambda e: e[5:], bands))

    with open(meta_path) as in_file:
        xml = in_file.read()
        a = xmltodict.parse(xml)
        download_list = list(
            a[xml_key]['n1:General_Info']['Product_Info']['Product_Organisation']['Granule_List']['Granule'][
                'IMAGE_FILE'])
    list_path = []
    if level == 1:
        for band in bands:
            band_path = get_path_band(band, download_list)
            list_path.append(band_path)
    else:
        for band in bands:
            suffix = "{}_{}".format(band.split('/')[-1], band.split('/')[0][1:])
            band_path = get_path_band(suffix, download_list)
            list_path.append(band_path)

    result = list(map(lambda e: '{}/{}.jp2'.format(path, e), list_path))

    gdal_merge = "{}/gdal_merge.py".format(GDAL_BIN)
    PYTHON_ENV_PATH = CONDA_PREFIX + "/bin/python"
    abs_python_path = PYTHON_ENV_PATH

    # merge files
    files_string = " ".join(result)
    merge_command = '{} {} -separate -a_nodata 0 -o {} '.format(abs_python_path, gdal_merge,
                                                                os.path.abspath(output_file)) + files_string
    os.system(merge_command)

def get_bands(code, product):
    if code == '10M':
        # ['B02', 'B03', 'B04', 'B08']
        return list(range(1, 5))
    if code == '20M':
        if product == 'L1C':
            # ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
            return list(range(1, 7))
        else:
            # ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'AOT', 'SCL', 'WVP']
            return list(range(1, 13))
    if code == '60M':
        if product == 'L1C':
            # ['B01', 'B09', 'B10']
            return list(range(1, 4))
        else:
            # ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B09', 'B11', 'B12', 'AOT', 'SCL', 'WVP']
            return list(range(1, 15))
    raise Exception(f"No support spatial resolution {code} with {product}")
    
def get_path_band(suffix, list_path):
    for path in list_path:
        if suffix in path:
            return path
        
def _translate(src_path, dst_path, profile="deflate", profile_options={}, **options):
    output_profile = cog_profiles.get(profile)
    output_profile.update(dict(BIGTIFF="IF_SAFER"))
    output_profile.update(profile_options)
    config = dict(
        GDAL_NUM_THREADS="ALL_CPUS",
        GDAL_TIFF_INTERNAL_MASK=True,
        GDAL_TIFF_OVR_BLOCKSIZE="128",
    )
    cog_translate(src_path,
                    dst_path,
                    output_profile,
                    config=config,
                    quiet=True,
                    add_mask=False,
                    **options,
                 )
    return True

def compute_bound(img_path):
    data = gdal.Open(img_path, GA_ReadOnly)
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

    geometry = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [tar_point_list]
                }
            }
        ]
    }
    data = None
    return geometry

def stretch_v2(path, output, min=2, max=98, nodata=0):
    abs_path = path
    with COGReader(abs_path) as cog:
        res = []
        try:
            stats = cog.stats(pmax=max, pmin=min, nodata=nodata)
            for _, value in stats.items():
                res.append({
                    'p2': value['pc'][0],
                    'p98': value['pc'][1],
                })
        except Exception:
            img = cog.preview()
    data = {
        'stretches': res
    }
    with open(output, 'w') as outfile:
        json.dump(data, outfile)

def write_json_file(data, name):
    with open(name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return name

def simplify_polygon(AOI_GEOMETRY):
    for d in AOI_GEOMETRY:
        d['geometry'] = shape(d['geometry'])

    gdf = gpd.GeoDataFrame(AOI_GEOMETRY).set_geometry('geometry')
    gdf.head()

    return AOI_GEOMETRY