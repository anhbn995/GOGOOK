import os
from task import celery
from lib.docker_executor import DockerExecutor, task_wrapper
from config.storage import ROOT_DATA_FOLDER
import json
import shutil
from lib.celery.mosaic.mosaic_task import MosaicTask
from lib.celery.mosaic.search_and_clip_task import SearchAndClipTask
from lib.celery.mosaic.search_images_task import SearchImagesTask
from lib.mosaic.utils import get_image_stats
from lib.indices_monitor.readers.aws.sentinel2 import S2COGReader
from rio_tiler.models import ImageData
import rasterio
from lib.indices_monitor.processors.sentinel2 import Sentinel2
from shapely.geometry import shape
from shapely.geometry.polygon import Polygon

@celery.task(bind=True, name="tasks.mosaic.search_images", track_started=True, acks_late=True, base=SearchImagesTask)
@task_wrapper
def search_images(self, source, params, task_detail, caller_id):
    executor = DockerExecutor("ssr/ssr-data-processing", self.request.id)
    executor.add_mount(ROOT_DATA_FOLDER, ROOT_DATA_FOLDER)
    executor.set_cmd([
        "python",
        "processors/discovery.py",
        source,
        '--aoi', params["aoi"],
        '--from-date', params["from_date"],
        '--to-date', params["to_date"],
        '--cloud-cover', str(params["cloud_cover"]
                             [0]), str(params["cloud_cover"][1])
    ])
    executor.run_container()
    lines = executor.logs
    lines = json.loads(''.join(lines[lines.index("Image Results:")+1:]))
    if(len(lines) < 1):
        raise Exception("No image was found by given data")
    return lines

@celery.task(bind=True, name="tasks.mosaic.search_and_clip_image", track_started=True, acks_late=True, base=SearchAndClipTask)
@task_wrapper
def search_and_clip_image(self, source, params, task_detail, caller_id):
    executor = DockerExecutor("ssr/ssr-data-processing", self.request.id)
    executor.add_mount(ROOT_DATA_FOLDER, ROOT_DATA_FOLDER)
    executor.set_cmd([
        "python",
        "processors/discovery.py",
        source,
        '--aoi', params["aoi"],
        '--from-date', params["from_date"],
        '--to-date', params["to_date"],
        '--cloud-cover', str(params["cloud_cover"]
                             [0]), str(params["cloud_cover"][1])
    ])
    executor.run_container()
    lines = executor.logs
    lines = json.loads(''.join(lines[lines.index("Image Results:")+1:]))
    if(len(lines) < 1):
        raise Exception("No image was found by given data")
    
    with open(params["aoi"], 'r') as f:
        geojson = json.load(f)
    geometry = geojson.get('features')[0].get('geometry')
    polygon: Polygon = shape(geometry)
    selected_image = ''
    min_cloud_rate = 1
    for image in lines:
        processor = Sentinel2(image, 'NDVI', polygon)
        cloud_rate = processor.get_cutline_info().get('field_cloud_cover')
        if(cloud_rate < min_cloud_rate):
            min_cloud_rate = cloud_rate
            selected_image = image
    print(f"Selected {selected_image} out of {len(lines)} images with cloud rate of {min_cloud_rate}")
    abso_path = f'/data/{task_detail.get("user_id")}/{task_detail.get("project_id")}/{caller_id}.tif'
    relat_path = f'{ROOT_DATA_FOLDER}{abso_path}'
    with S2COGReader(selected_image) as sen2:
        data = sen2.feature(shape=geometry, bands=('B02','B03','B04','B08'), vrt_options={'CUTLINE_ALL_TOUCHED': True})
        nodata = sen2.info(shape=geometry, bands=('B02','B03','B04','B08'), vrt_options={'CUTLINE_ALL_TOUCHED': True}).nodata_type
        file_from_image_data(data, relat_path, 4, 0)
    stats = get_image_stats(relat_path)
    self.return_value = {
        'path': abso_path,
        'stats': stats
    }
    print("Finished!")

@celery.task(bind=True, name="tasks.mosaic.download", track_started=True, acks_late=True, base=SearchImagesTask)
@task_wrapper
def download(job_id, image_ids, task_detail, caller_id):
    return download_handle(job_id, image_ids)

@celery.task(bind=True, name="tasks.mosaic.mosaic", track_started=True, acks_late=True, base=MosaicTask)
@task_wrapper
def mosaic(self, output_path, download_job_id, image_ids, group_by_level, cutline_path, task_detail, caller_id):
    print("Start downloading images")
    images_data_folder, raw_image_dir = download_handle(download_job_id, image_ids)
    print("Start mosaic 10m")
    mosaic_handle(self.request.id, raw_image_dir, 10,
           group_by_level, output_path, f'{images_data_folder}/images.json', cutline_path)

    abso_path = f'/data/{task_detail.get("user_id")}/{task_detail.get("project_id")}/{caller_id}.tif'
    relat_path = f'{ROOT_DATA_FOLDER}{abso_path}'
    move_mosaic_result(output_path, relat_path)
    stats = get_image_stats(relat_path)
    self.return_value = {
        'path': abso_path,
        'stats': stats
    }
    print("Finished!")

def download_handle(job_id, image_ids):
    raw_image_dir = f'{ROOT_DATA_FOLDER}/raw_images'
    output_dir = f'{ROOT_DATA_FOLDER}/tmp/{job_id}'
    os.makedirs(output_dir, exist_ok=True)
    images_path = f'{output_dir}/images.json'
    with open(images_path, 'w') as f:
        json.dump(image_ids, f)
    executor = DockerExecutor("ssr/ssr-data-processing", job_id)
    executor.add_mount(ROOT_DATA_FOLDER, ROOT_DATA_FOLDER)
    executor.set_cmd([
        "python",
        "processors/download.py",
        '--data-folder', raw_image_dir,
        images_path
    ])
    executor.run_container()
    update_downloaded_image_list(raw_image_dir, images_path)
    return output_dir, raw_image_dir

def mosaic_handle(job_id, data_folder, resolution, group_by_level, output_path, images_path, cutline_path=None):
    executor = DockerExecutor("ssr/ssr-data-processing", job_id)
    executor.add_mount(ROOT_DATA_FOLDER, ROOT_DATA_FOLDER)
    cmd = [
        "python",
        "processors/mosaic.py",
        '--resolution', str(resolution),
        '--output', output_path
    ]
    if group_by_level:
        cmd += ['--group-by', str(group_by_level)]
    if cutline_path:
        cmd += ['--cutline-path', cutline_path]
    if data_folder:
        cmd += ['--data-folder', data_folder]
    cmd.append(images_path)
    executor.set_cmd(cmd)
    executor.run_container()

def update_downloaded_image_list(raw_image_dir, images_path):
    f = open(images_path)
    searched_images = json.load(f)
    downloaded_images = searched_images.copy()
    print("searched images: ", searched_images)
    all_images = os.listdir(raw_image_dir)
    print("All images: ", all_images)
    for image in searched_images:
        if f'{image}.SAFE' not in all_images:
            downloaded_images.remove(image)
    print("downloaded images: ", downloaded_images)
    with open(images_path, 'w') as f:
        json.dump(downloaded_images, f)

def move_mosaic_result(source_dir, des):
    src_directory = os.fsencode(source_dir)
    images = os.listdir(src_directory)
    if(len(images) < 1):
        raise Exception("No result was found.")
    elif(len(images) > 1):
        raise Exception(f"There are multiple results in: {source_dir}.")
    result_src_path = f'{source_dir}/{os.fsdecode(images[0])}'
    print(f"Found a result in: {result_src_path}, moving it to result storage")
    file_name =  os.path.basename(des)
    des_dir = os.path.dirname(des)
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    shutil.move(result_src_path, f'{des_dir}/{file_name}')

def file_from_image_data(data: ImageData, store_path, band_count, nodata=0):
    profile = {
        'driver': 'GTiff',
        'dtype': data.data.dtype,
        'nodata': nodata,
        'width': data.width,
        'height': data.height,
        'count': band_count,
        'crs': data.crs,
        'transform': data.transform,
    }
    with rasterio.open(store_path, 'w', **profile) as dst:
        dst.write(data.data)