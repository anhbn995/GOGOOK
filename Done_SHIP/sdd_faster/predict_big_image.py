import pathlib
import threading
import numpy as np
import tensorflow as tf
import concurrent.futures
from tqdm import tqdm

import glob, os
import rasterio
import geopandas as gpd
from rasterio.windows import Window
from shapely.geometry import Polygon

from object_detection.utils import config_util
from object_detection.builders import model_builder

def write_window_many_chanel(output_ds, arr_c, window_draw_pre):
    s_h, e_h ,s_w, e_w, sw_w, sw_h, size_w_crop, size_h_crop = window_draw_pre 
    output_ds.write(arr_c[s_h:e_h,s_w:e_w],window = Window(sw_w, sw_h, size_w_crop, size_h_crop), indexes = 1)


def read_window_and_index_result(crop_size, h_crop_start, w_crop_start, start_w_org, start_h_org, padding, h, w, tmp_img_size_model, src_img, num_band_train):
    """
        Trả về img de predict vs kich thước model
        Và vị trí để có thể ghi mask vào trong đúng vị trí ảnh
    """
    if h_crop_start < 0 and w_crop_start < 0:
        h_crop_start = 0
        w_crop_start = 0
        size_h_crop = crop_size + padding
        size_w_crop = crop_size + padding
        img_window_crop  = src_img.read([*range(1, num_band_train+1)],window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
        tmp_img_size_model[:, padding:, padding:] = img_window_crop
        window_draw_pre = [padding, crop_size + padding, padding, crop_size + padding, start_w_org, start_h_org, crop_size, crop_size]

    elif h_crop_start < 0:
        h_crop_start = 0
        size_h_crop = crop_size + padding
        size_w_crop = min(crop_size + 2*padding, w - start_w_org + padding)
        img_window_crop  = src_img.read([*range(1, num_band_train+1)],window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
        if size_w_crop == w - start_w_org + padding:
            end_c_index_w =  size_w_crop
            tmp_img_size_model[:,padding:,:end_c_index_w] = img_window_crop
        else:
            end_c_index_w = crop_size + padding
            tmp_img_size_model[:, padding:,:] = img_window_crop
        window_draw_pre = [padding, crop_size + padding ,padding, end_c_index_w, start_w_org, start_h_org,  min(crop_size, w - start_w_org), crop_size]

    elif w_crop_start < 0:
        w_crop_start = 0
        size_w_crop = crop_size + padding
        size_h_crop = min(crop_size + 2*padding, h - start_h_org + padding)
        img_window_crop  = src_img.read([*range(1, num_band_train+1)],window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
        if size_h_crop == h - start_h_org + padding:
            end_c_index_h =  size_h_crop
            tmp_img_size_model[:,:end_c_index_h,padding:] = img_window_crop
        else:
            end_c_index_h = crop_size + padding
            tmp_img_size_model[:,:, padding:] = img_window_crop
        window_draw_pre = [padding, end_c_index_h, padding, crop_size + padding, start_w_org, start_h_org, crop_size, min(crop_size, h - start_h_org)]
    
    else:
        size_w_crop = min(crop_size +2*padding, w - start_w_org + padding)
        size_h_crop = min(crop_size +2*padding, h - start_h_org + padding)
        img_window_crop  = src_img.read([*range(1, num_band_train+1)],window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
        if size_w_crop < (crop_size + 2*padding) and size_h_crop < (crop_size + 2*padding):
            end_c_index_h = size_h_crop
            end_c_index_w = size_w_crop
            tmp_img_size_model[:,:end_c_index_h,:   end_c_index_w] = img_window_crop
        elif size_w_crop < (crop_size + 2*padding):
            end_c_index_h = crop_size + padding
            end_c_index_w = size_w_crop
            tmp_img_size_model[:,:,:end_c_index_w] = img_window_crop
        elif size_h_crop < (crop_size + 2*padding):
            end_c_index_w = crop_size + padding
            end_c_index_h = size_h_crop
            tmp_img_size_model[:,:end_c_index_h,:] = img_window_crop
        else:
            end_c_index_w = crop_size + padding
            end_c_index_h = crop_size + padding
            tmp_img_size_model[:,:,:] = img_window_crop
        window_draw_pre = [padding, end_c_index_h, padding, end_c_index_w, start_w_org, start_h_org, min(crop_size, w - start_w_org), min(crop_size, h - start_h_org)]
    return tmp_img_size_model, window_draw_pre, Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop)



def predict_win(numpy_chanel_first, detect_fn):
    image_np_chanel_last = np.transpose(numpy_chanel_first, (1,2,0))
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np_chanel_last, 0), dtype=tf.float32)
    detections, _, _ = detect_fn(input_tensor)
    return detections


def convert_detections_to_polygon(detections, source_data, windo, min_score_thresh):
    transfrom_win = source_data.window_transform(windo)
    im_width, im_height = windo.width, windo.height
    all_boxes = detections['detection_boxes'][0].numpy()
    all_scores = detections['detection_scores'][0].numpy()
    
    list_polygons = list()
    list_scores = list()
    for i in range(all_boxes.shape[0]):
        if all_scores[i] > min_score_thresh:
            ymin, xmin, ymax, xmax = tuple(all_boxes[i].tolist())
            (left_pixel, right_pixel, top_pixel, bottom_pixel) = (xmin * im_width, xmax * im_width,
                                    ymin * im_height, ymax * im_height)
            
            left_geo, top_geo = transfrom_win * (left_pixel, top_pixel)
            right_geo, bottom_geo = transfrom_win * (right_pixel, bottom_pixel)
            polygon = Polygon([(left_geo, top_geo), (right_geo, top_geo), (right_geo, bottom_geo), (left_geo, bottom_geo)])
            list_polygons.append(polygon)
            list_scores.append(all_scores[i])
    return list_polygons, list_scores
 
def predict_lager(fp_img, model_ship, model_size, crop_size, score_thresh):
    num_band_train = 3
    with rasterio.open(fp_img) as src:
        h,w = src.height,src.width
        source_crs = src.crs
        source_transform = src.transform
    
    padding = int((model_size - crop_size)/2)
    list_weight = list(range(0, w, crop_size))
    list_hight = list(range(0, h, crop_size))
    
    list_polygons_all = list()
    list_scores_all = list()
    
    with tqdm(total=len(list_hight)*len(list_weight)) as pbar:
        with rasterio.open(fp_img) as src:
            for start_h_org in list_hight:
                for start_w_org in list_weight:
                    # vi tri bat dau
                    h_crop_start = start_h_org - padding
                    w_crop_start = start_w_org - padding
                    
                    # kich thuoc
                    tmp_img_model = np.zeros((num_band_train, model_size,model_size))
                    tmp_img_model, _, wind = read_window_and_index_result(crop_size, h_crop_start, w_crop_start, start_w_org, start_h_org, padding, h, w, tmp_img_model, src, num_band_train)
                    detections = predict_win(tmp_img_model, detect_fn)
                    list_polygons, list_scores = convert_detections_to_polygon(detections, src, wind, score_thresh)
                    list_polygons_all += list_polygons
                    list_scores_all +=list_scores
                    pbar.update()
    return list_polygons_all, list_scores_all, source_crs
                    

def NMS_polygons(list_polygons, list_scores, list_labels=None, iou_threshold=0.2):
    list_shapely_polygons = [Polygon(polygon) for polygon in list_polygons]
    list_bound = [np.array(polygon.bounds) for polygon in list_shapely_polygons]
    indexes = tf.image.non_max_suppression(np.array(list_bound), np.array(list_scores), len(list_scores),iou_threshold=iou_threshold)
    result_polygons = [list_polygons[idx] for idx in indexes]
    result_scores = [list_scores[idx] for idx in indexes]
    if list_labels:
        result_labels = [list_labels[idx] for idx in indexes]
        return result_polygons, result_scores, result_labels
    else:
        return result_polygons, result_scores


def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


def main(fp_img, detect_fn, model_size, crop_sizes, fp_out_shape):
    
    # load model is the best
    # filenames = glob.glob(os.path.join(dir_weight,'*.index'))
    # filenames = list(pathlib.Path(dir_weight).glob('*.index'))
    # filenames.sort()
    # print(filenames)
    # # #recover our saved model
    # model_dir = dir_weight
    # #generally you want to put the last ckpt from training in here
    # configs = config_util.get_configs_from_pipeline_file(pipeline_file)
    # model_config = configs['model']
    # detection_model = model_builder.build(model_config=model_config, is_training=False)
    # # print(detection_model)
    # # Restore checkpoint
    # ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    # print(os.path.join(str(filenames[-1]).replace('.index','')))
    # ckpt.restore(os.path.join(str(filenames[-1]).replace('.index','')))
    # detect_fn = get_model_detection_function(detection_model)
    
    
    # predict each win
    list_polygons_all, list_scores_all, source_crs = predict_lager(fp_img, detect_fn, model_size, crop_sizes, score_thresh=0.3)
    gdf = gpd.GeoDataFrame(geometry=list_polygons_all)
    gdf.crs = source_crs
    gdf.to_file(fp_out_shape)
    
    result_polygons, result_scores = NMS_polygons(list_polygons_all, list_scores_all, list_labels=None, iou_threshold=0.2)
    gdf2 = gpd.GeoDataFrame(geometry=result_polygons)
    gdf2.crs = source_crs
    gdf2.to_file(fp_out_shape.replace('.shp', '_nms.shp'))
    # pass

    
    
    
if __name__=="__main__":
    import time
    x = time.time()
    fp_img = r'/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/Ship/STCD.tif'
    dir_weight = r'/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/code/Try/SSD/train_custom/ssd_resnet50_v1_fpn_640x640_coco17_tpu'
    pipeline_file = r'/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/code/Try/SSD/export_model_custom/ssd_resnet50_v1_fpn_640x640_coco17_tpu/pipeline.config'
    model_size = 128
    crop_sizes = 100
    
    dir_out = f'/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/Ship/RS/size_{model_size}/ssd'
    os.makedirs(dir_out, exist_ok=True)
    fp_out_shape = os.path.join(dir_out, f'STCD_{os.path.basename(dir_weight)}.shp')
    
    filenames = list(pathlib.Path(dir_weight).glob('*.index'))
    filenames.sort()
    print(filenames)
    # #recover our saved model
    model_dir = dir_weight
    #generally you want to put the last ckpt from training in here
    configs = config_util.get_configs_from_pipeline_file(pipeline_file)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)
    # print(detection_model)
    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    print(os.path.join(str(filenames[-1]).replace('.index','')))
    ckpt.restore(os.path.join(str(filenames[-1]).replace('.index','')))
    detect_fn = get_model_detection_function(detection_model)
    
    main(fp_img, detect_fn, model_size, crop_sizes, fp_out_shape)
    total_minutes = (time.time() - x)/60
    minutes = int(total_minutes)
    decimal_part = total_minutes - minutes
    seconds = int(decimal_part * 60)
    
    with open(fp_out_shape.replace('.shp', '.txt'), 'w') as file:
    # Ghi nội dung vào tệp tin
        file.write(f'Thoi gian cua mo hinh predict la : {str(minutes)}m{str(seconds)}s')
    
    