import os
import glob
import uuid
import copy
import json
import fiona
import shutil
import urllib
import datetime
import requests
import rasterio
import numpy as np
import multiprocessing
import rasterio.mask
import geopandas as gp

from multiprocessing.pool import Pool
from functools import partial

from shapely.ops import cascaded_union, unary_union
from shapely.geometry import shape

from tqdm import tqdm
from skyamqp import AMQP_Client
from shapely.geometry import Polygon, mapping
from download_and_processing_cloud.greencover_api.download_utils.utils import last_day_of_month, simplify_polygon, make_nas_temp_folder_in_root_data_folder,\
    get_raw_image, process_sentinel2_pci, get_bands_for_gdal, gdal_merge_file, \
    clip_image, _translate, compute_bound, stretch_v2, write_json_file

os.environ['AMQP_HOST'] = '192.168.4.100'
os.environ['AMQP_PORT'] = '5672'
os.environ['AMQP_VHOST'] = '/eof'
os.environ['AMQP_USERNAME'] = 'eof_rq_worker'
os.environ['AMQP_PASSWORD'] = '123'
connection = AMQP_Client(
    host=os.environ.get('AMQP_HOST'),
    port=os.environ.get('AMQP_PORT'),
    virtual_host=os.environ.get('AMQP_VHOST'),
    username=os.environ.get('AMQP_USERNAME'),
    password=os.environ.get('AMQP_PASSWORD'),
    heartbeat=5
)
print('Connected AMQP host!!')


def cut(img_name, img_dir, box_dir, img_cut_dir):
    image_path = os.path.join(img_dir, img_name+'.jp2')
    shape_path = glob.glob(os.path.join(box_dir, '*.shp'))[0]

    with rasterio.open(image_path, mode='r+') as src:
        projstr = src.crs.to_string()
        # print(projstr)
        check_epsg = src.crs.is_epsg_code
        if check_epsg:
            epsg_code = src.crs.to_epsg()
            # print(epsg_code)
        else:
            epsg_code = None
    if epsg_code:
        out_crs = {'init': 'epsg:{}'.format(epsg_code)}
    else:
        out_crs = projstr
    bound_shp = gp.read_file(shape_path)
    bound_shp = bound_shp.to_crs(out_crs)

    for index2, row_bound in bound_shp.iterrows():
        geoms = row_bound.geometry
        img_cut = img_name+"_{}.tif".format(index2)
        img_cut_path = os.path.join(img_cut_dir, img_cut)
        # if os.path.exists(img_cut_path):
        #     pass
        # else:
        try:
            if not os.path.exists(img_cut_path):
                with rasterio.open(image_path, BIGTIFF='YES') as src:
                    out_image, out_transform = rasterio.mask.mask(
                        src, [geoms], crop=True)
                    out_meta = src.meta
                # print( "height",out_image.shape[1])
                out_meta.update({"driver": "GTiff",
                                 "height": out_image.shape[1],
                                 "width": out_image.shape[2],
                                 "transform": out_transform})
                with rasterio.open(img_cut_path, "w", **out_meta) as dest:
                    dest.write(out_image)
        except:
            pass
            # raise Exception("Error crop image")


def create_list_id_2(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.jp2"):
        list_id.append(file[:-4])
    return list_id


def main_cut_img_2(img_path, box_path, tmp_path):
    core = multiprocessing.cpu_count()//4
    img_list = create_list_id_2(img_path)

    img_cut_dir = tmp_path+'_cut'

    print("Run crop image with aoi ...")
    if not os.path.exists(img_cut_dir):
        os.makedirs(img_cut_dir)

    p_cnt = Pool(processes=core)
    p_cnt.map(partial(cut, img_dir=img_path, box_dir=box_path,
                      img_cut_dir=img_cut_dir), img_list)
    p_cnt.close()
    p_cnt.join()
    print("Done")
    return img_cut_dir


def get_list_workspace(input_url, token):
    list_workspace = []
    workspace = requests.get(input_url, headers={'Authorization': token})
    if not workspace.ok:
        raise Exception("Can't connect, please check your token.")
    f = workspace.json()
    for i in f['data']:
        list_workspace.append([i['id'], i['name']])
    return list_workspace


def search_img_url_v2(AOI, query):
    list_id = []
    AOI_1 = copy.deepcopy(AOI)
    AOI_GEO = simplify_polygon(AOI_1)
    for n, i in enumerate(AOI_GEO):
        item = i['geometry'].simplify(1)
        url = f"https://finder.creodias.eu/resto/api/collections/Sentinel2/search.json?geometry={item}&" + urllib.parse.urlencode(
            query, doseq=False, safe='[,]', quote_via=urllib.parse.quote)
        response = requests.get(url.replace('%20', ''))
        # print(url)
        if response.ok:
            find_list = response.json()
            _a = find_list['properties']['links'][0]['href']
            response = requests.get(_a)
            if response.ok:
                #                 print(n , response.json()['properties']['totalResults'])
                for j in response.json()['features']:
                    name_img = j['properties']['title'].replace('.SAFE', '')
                    date_time = j['properties']['completionDate']
                    if [name_img, date_time] not in list_id:
                        list_id.append([name_img, date_time])
    return list_id


def filter_follow_cloud(dir_path, outpath, shape_path, threshhold):
    name = '_SCL_20m'
    # print(dir_path)
    granule_folder = os.path.join(os.path.dirname(dir_path), 'GRANULE')
    band20m_folder = os.path.join(granule_folder, os.listdir(
        granule_folder)[0], 'IMG_DATA', 'R20m')

    for file in glob.glob(os.path.join(band20m_folder, '*.jp2')):
        if name in file:
            band_scl_file = file
            break

    tmp_band_scl_folder = band20m_folder.replace('R20m', 'R20m_tmp')
    if not os.path.exists(tmp_band_scl_folder):
        os.mkdir(tmp_band_scl_folder)
    shutil.copyfile(band_scl_file, os.path.join(
        tmp_band_scl_folder, os.path.basename(band_scl_file)))
    crop_aoi_img_folder = main_cut_img_2(
        tmp_band_scl_folder, shape_path, tmp_band_scl_folder)
    crop_aoi_img_file = glob.glob(
        os.path.join(crop_aoi_img_folder, '*.tif'))[0]
    print(os.path.exists(crop_aoi_img_file))
    img = rasterio.open(crop_aoi_img_file).read()

    value, count = np.unique(img, return_counts=True)
    total_pixel = 0
    cloud_pixel = 0
    haze_pixel = 0
    for n, i in enumerate(value):
        if i != 0:
            total_pixel += count[n]

        if i == 8:
            cloud_pixel += count[n]
            haze_pixel += count[n]
        elif i == 9:
            cloud_pixel += count[n]
        else:
            pass

    percent_cloud = (cloud_pixel/total_pixel) * 100
    percent_haze = (haze_pixel/total_pixel) * 100
    if percent_cloud < (threshhold*100):
        classification_path = os.path.join(
            outpath, 'cloud_<_%s%%' % str(threshhold*100))
        if not os.path.exists(classification_path):
            os.mkdir(classification_path)
    else:
        classification_path = os.path.join(
            outpath, 'cloud_>_%s%%' % str(threshhold*100))
        if not os.path.exists(classification_path):
            os.mkdir(classification_path)
    return classification_path, percent_haze


def download_image_v2(image_id_l1c, out_path, shape_path, level=2, code='10M', product='L2A',
                      LICENSE_PCI=True, has_hazerem=False, geom=None, threshhold=0.7):
    print("Prepare to download image...")
    attempts = 0
    file_id = uuid.uuid4()

    print("Get list id senl2a image...")

    img_id_date = image_id_l1c
    image_id = img_id_date[0]
    date_img = str(int((img_id_date[1]).split('-')[1]))
    name_month = 'T%s' % (date_img)
    name_year = str(int((img_id_date[1]).split('-')[0]))
    name_month_year = name_month+'-'+name_year
    out_folder = os.path.join(out_path, name_month_year)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    newfile_1 = os.path.join(out_folder, 'cloud_<_%s%%' %
                             str(threshhold*100), image_id+'.tif')
    newfile_2 = os.path.join(out_folder, 'cloud_>_%s%%' %
                             str(threshhold*100), image_id+'.tif')

    if os.path.exists(newfile_1):
        newfile = newfile_1
        pass
    elif os.path.exists(newfile_2):
        newfile = newfile_2
        pass
    else:
        if LICENSE_PCI:
            temp_dir = make_nas_temp_folder_in_root_data_folder()
        print("Done")

        print("Download senl2a image...")
        download_utils_rpcClient = connection.create_RPC_Client(
            'download-image-utils')
        resp = download_utils_rpcClient.send('sentinel2.download_from_aws', {
                                             'id': image_id, 'level': level})
        if not resp['success']:
            resp = download_utils_rpcClient.send(
                'sentinel2.download_sentinel_hub', {'id': image_id})
            if not resp['success']:
                raise Exception('Please download at Imagery Guru !')

        raw_image = get_raw_image(resp)
        resp_download = resp

        dir_path = '/home/geoai/geoai_data_test2/awsdata/sen2/%s.SAFE/tmp' % (
            image_id)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        out_path = "%s/%s.tif" % (dir_path, file_id.hex)
        out_cls_path, percent_haze = filter_follow_cloud(
            dir_path, out_folder, shape_path, threshhold)
        newfile = os.path.join(out_cls_path, image_id +
                               '_haze_%s.tif' % str(int(percent_haze)))
        # out_path = os.path.join(out_cls_path, file_id.hex)
        pre_out_path = "%s/%s.tif" % (temp_dir, file_id.hex)
        print("Finished download senl2a(raw)")

        print("Try to run PCI...")
        if LICENSE_PCI:
            try:
                attempts = process_sentinel2_pci(attempts, temp_dir, product, code, resp_download['save_path'],
                                                 pre_out_path, has_hazerem)
                if attempts == 0:
                    print("attempts: %s," % str(attempts), "Finished run PCI.")
            except Exception as e:
                print(e)
                pass

        stretch_path = "%s/%s.json" % (dir_path, file_id.hex)
        if attempts > 3 or not LICENSE_PCI:
            print("Can't run PCI, change to run gdal...")
            gdal_merge_file(resp_download['save_path'], pre_out_path, get_bands_for_gdal(
                code, product), level)
            print("Finished run gdal.")

        if geom:
            print("Clip image with AOI...")
            computed_geom = geom
            clip_image(geom, pre_out_path, str(file_id), dir_path, temp_dir)
            print("Finished clip image.")
        else:
            print("Translate cog...")
            _translate(pre_out_path, out_path)
            computed_geom = compute_bound(out_path)
            print("Done")

        print("Stretch image...")
        stretch_v2(out_path, stretch_path)
        print("Done")
        rename_output = out_path.replace(
            os.path.basename(out_path), image_id+'.tif')
        os.rename(out_path, rename_output)
        shutil.copyfile(rename_output, newfile)
        shutil.rmtree(temp_dir)
        # shutil.rmtree('/home/geoai/geoai_data_test2/awsdata/sen2/%s.SAFE'%(image_id)) # TO DO
        # print("Finished all.")
    return newfile, out_folder


def create_shp(box_aoi, box_path, name_shp):
    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int'},
    }
    box_aoi_path = os.path.join(box_path, name_shp)
    print(box_aoi_path)
    geoms = [shape(feature['geometry']) for feature in box_aoi]
    box_aoi_merge = unary_union(geoms)
    if not os.path.exists(box_aoi_path):
        with fiona.open(box_aoi_path, 'w', 'ESRI Shapefile', schema, crs='EPSG:4326') as c:
            c.write({
                'geometry': mapping(box_aoi_merge),
                'properties': {'id': 0},
            })


def main_download(folder_path, name_ws, name_aoi, input_url, token, month, years, start_date, end_date, CLOUD_COVER, data, json_path, threshhold):
    print("Create workspace...")
    correct_ws = False
    correct_aoi = False
    list_month_folder = []
    list_workspace = get_list_workspace(input_url, token)
    for i in list_workspace:
        if name_ws in i:
            correct_ws = True
            id_workspace = i[0]
            name_workspace = i[1]
            workspace_path = os.path.join(folder_path, name_workspace)
            if not os.path.exists(workspace_path):
                os.makedirs(workspace_path)

    if not CLOUD_COVER:
        cloudCover = [0, 100]
    else:
        cloudCover = [0, int(CLOUD_COVER*100)]

    if not correct_ws:
        raise Exception("Incorrect name workspace.")
    print("Done")

    print("Get AOI...")
    aoi_url = "https://api-aws.eofactory.ai/api/workspaces/%s/aois?region=sea" % (
        id_workspace)
    aoi_geo = requests.get(aoi_url, headers={'Authorization': token})
    AOI = aoi_geo.json()

    for j in AOI['data']:
        if name_aoi == j['name']:
            correct_aoi = True
            GEOMETRY = j['geom']['features'][0]['geometry']
            AOI_GEOMETRY = []
            for l in j['geom']['features']:
                AOI_GEOMETRY.append(l)

    if not correct_aoi:
        raise Exception("Incorrect name AOI.")
    print("Done")

    if os.path.exists(json_path):
        f = open(json_path)
        data = json.load(f)
    else:
        raise Exception("File json isn't exists")

    if len(data['AOI']) == 0:
        data.update({'AOI': AOI_GEOMETRY})

    if data['workspace_path'] != workspace_path:
        data.update({'workspace_path': workspace_path})

    write_json_file(data, json_path)
    box_aoi = data['AOI']
    name_shp = "%s.shp" % (str(name_ws))
    box_path = os.path.join(workspace_path, 'box')
    if not os.path.exists(box_path):
        os.makedirs(box_path)
    create_shp(box_aoi, box_path, name_shp)

    for year in years:
        for i in month:
            try:
                print(data['list_image']['%s' % (i)])
            except:
                data['list_image'].update({'%s' % (i): []})

            if len(data['list_image']) == 0:
                data['list_image'].update({'%s' % (i): []})
                id_image = []
            else:
                if data['list_image']['%s' % (i)] == None:
                    id_image = []
                else:
                    id_image = data['list_image']['%s' % (i)]

            print("Download image T%s-%s..." % (str(i), str(year)))
            end_date = int(str(last_day_of_month(
                datetime.date(year, i, start_date))).split('-')[-1])
            query = {
                'status': 'all',
                'dataset': 'ESA-DATASET',
                'maxRecords': 1000,
                'sortParam': 'startDate',
                'sortOrder': 'descending',
                'startDate': str(datetime.date(year, i, start_date)),
                'completionDate': str(datetime.date(year, i, end_date)),
                'processingLevel': 'LEVEL2A',
                'cloudCover': cloudCover
            }
            list_img = search_img_url_v2(AOI=AOI_GEOMETRY, query=query)

            if len(list_img) == 0:
                raise Exception("Please enter a valid API key")

            for j in tqdm(list_img):
                image_id_l1c = j
                if image_id_l1c not in id_image:
                    download_folder = os.path.join(
                        workspace_path, 'img_origin')
                    if not os.path.exists(download_folder):
                        os.mkdir(download_folder)
                    newfile, out_folder = download_image_v2(
                        image_id_l1c, download_folder, box_path, threshhold=threshhold)
                    if out_folder not in list_month_folder:
                        list_month_folder.append(out_folder)
                    id_image.append(os.path.basename(newfile))
                    print('\n')
                    data['list_image'].update({'%s' % (str(i)): id_image})
                    write_json_file(data, json_path)

            print("Done")
    return workspace_path, list_month_folder


def get_token(login_url, user_info):
    login_info = requests.post(login_url, json=user_info)
    if login_info.ok:
        infomation = login_info.json()
        token = 'Bearer ' + infomation['body']['token']
    else:
        raise Exception("Can't get token, please check user id")
    return token


def main(folder_path, temp_path, name_ws, name_aoi, month, year, start_date, end_date, CLOUD_COVER):
    threshhold = 0.7
    login_url = "https://auth.eofactory.ai/login"
    input_url = 'https://api-aws.eofactory.ai/api/workspaces?region=sea'
    user_info = {"email": 'quyet.nn@eofactory.ai',
                 "password": 'Quyet135667799'}
    token = get_token(login_url, user_info)

    name_json_file = os.path.join(os.getcwd(), 'requirements.json')
    try:
        f = open(name_json_file)
        data = json.load(f)
    except:
        data = {'workspace_path': "",
                'temp_path': temp_path,
                'list_image': {},
                'weights': {},
                'AOI': []}
        write_json_file(data, name_json_file)

    workspace_path, list_month_folder = main_download(folder_path, name_ws, name_aoi, input_url, token, month, year,
                                                      start_date, end_date, CLOUD_COVER, data, name_json_file, threshhold)
    print(workspace_path, name_json_file)
    folder_download = os.path.join(
        workspace_path, 'download_list_%s_%s.json' % (str(month), str(year)))
    shutil.copyfile(name_json_file, folder_download)
    os.remove(name_json_file)
    print("Finished all")
    return workspace_path, list_month_folder


if __name__ == "__main__":
    month = [4]
    year = [2022]
    start_date = 1
    # end_date = None is download all
    end_date = None
    CLOUD_COVER = 1.0

    # Check before run
    name_aoi = "new_aoi"
    name_ws = "Green Cover Npark Singapore"
    temp_path = '/home/geoai/geoai_data_test2'
    folder_path = '/home/quyet/DATA_ML/Projects'
    workspace_path = main(folder_path, temp_path, name_ws,
                          name_aoi, month, year, start_date, end_date, CLOUD_COVER)
