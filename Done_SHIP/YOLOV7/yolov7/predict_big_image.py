import cv2
import pathlib
import numpy as np
from tqdm import tqdm

import glob, os
import rasterio
import geopandas as gpd
from rasterio.windows import Window
from shapely.geometry import Polygon

import cv2
import torch
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, scale_coords, set_logging, non_max_suppression
from utils.torch_utils import select_device, time_synchronized, TracedModel

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



def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



def load_model(opt):
    weights, imgsz = opt['weights'], opt['img-size']
    set_logging()
    device = select_device(opt['device'])
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()

    names = model.module.names if hasattr(model, 'module') else model.names  # tra ve ten lop ///
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]  # tra ve mau
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    return model, names

def predict_win(model, names, win_chanel_first, opt, transfrom_win):
    list_polygons = list()
    list_scores = list()
    
    
    with torch.no_grad():
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(opt['img-size'], s=stride)  # check img_size
        img0 = win_chanel_first.transpose(1,2,0)
        img = letterbox(img0, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        classes = None
        if opt['classes']:
            classes = []
            for class_name in opt['classes']:
                classes.append(names.index(class_name))

        pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=classes, agnostic=False)
        t2 = time_synchronized()
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                for *xyxy, conf, cls in reversed(det):
                    left_pixel, top_pixel, right_pixel, bottom_pixel = [tensor.item() for tensor in xyxy]
                    left_geo, top_geo = transfrom_win * (left_pixel, top_pixel)
                    right_geo, bottom_geo = transfrom_win * (right_pixel, bottom_pixel)
                    polygon = Polygon([(left_geo, top_geo), (right_geo, top_geo), (right_geo, bottom_geo), (left_geo, bottom_geo)])
                    list_polygons.append(polygon)
                    list_scores.append(conf.item())
        return list_polygons, list_scores
        

def predict_lager(fp_img, model, opt, model_size, crop_size):
    num_band_train = 3
    with rasterio.open(fp_img) as src:
        h,w = src.height,src.width
        source_crs = src.crs
    
    padding = int((model_size - crop_size)/2)
    list_weight = list(range(0, w, crop_size))
    list_hight = list(range(0, h, crop_size))
    
    list_polygons_all = list()
    list_scores_all = list()
    model, names = load_model(opt=opt)
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
                    list_polygons, list_scores = predict_win(model, names, tmp_img_model, opt,  src.window_transform(wind))
                    list_polygons_all += list_polygons
                    list_scores_all +=list_scores
                    pbar.update()
    return list_polygons_all, list_scores_all, source_crs
                    

# def NMS_polygons(list_polygons, list_scores, list_labels=None, iou_threshold=0.2):
#     list_shapely_polygons = [Polygon(polygon) for polygon in list_polygons]
#     list_bound = [np.array(polygon.bounds) for polygon in list_shapely_polygons]
#     indexes = tf.image.non_max_suppression(np.array(list_bound), np.array(list_scores), len(list_scores),iou_threshold=iou_threshold)
#     result_polygons = [list_polygons[idx] for idx in indexes]
#     result_scores = [list_scores[idx] for idx in indexes]
#     if list_labels:
#         result_labels = [list_labels[idx] for idx in indexes]
#         return result_polygons, result_scores, result_labels
#     else:
#         return result_polygons, result_scores



def main(fp_img, model, opt, model_size, crop_sizes, fp_out_shape):
    
    # predict each win
    list_polygons_all, list_scores_all, source_crs = predict_lager(fp_img, model, opt, model_size, crop_sizes)
    gdf = gpd.GeoDataFrame(geometry=list_polygons_all)
    gdf['score'] = list_scores_all
    gdf.crs = source_crs
    gdf.to_file(fp_out_shape)
    
    # result_polygons, result_scores = NMS_polygons(list_polygons_all, list_scores_all, list_labels=None, iou_threshold=0.2)
    # gdf2 = gpd.GeoDataFrame(geometry=result_polygons)
    # gdf2.crs = source_crs
    # gdf2.to_file(fp_out_shape.replace('.shp', '_nms.shp'))
    # pass

    
    
    
if __name__=="__main__":
    import time
    x = time.time()
    fp_img = r'/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/Ship/STCD.tif'
    model_size = 128
    crop_sizes = 100
    fp_weight = r"/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/code/YOLO/Custom/TrainYOLOv7/yolov7/runs/train/exp2/weights/best.pt"
    mydataset = r"/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/code/YOLO/Custom/TrainYOLOv7/yolov7/data/mydataset.yaml"
    
    dir_out = r'/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/Ship/RS/size_128/yolov7'
    os.makedirs(dir_out, exist_ok=True)
    fp_out_shape = os.path.join(dir_out, f'STCD_yolov7_size{model_size}.shp')
    opt  = {
    
            "weights": fp_weight, 
            "yaml"   : mydataset,
            "img-size": model_size, # default image size
            "conf-thres": 0.5, # confidence threshold for inference.
            "iou-thres" : 0.45, # NMS IoU threshold for inference.
            "device" : '0',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
            "classes" : None
        }
    
    # load model
    with torch.no_grad():
        weights, imgsz = opt['weights'], opt['img-size']
        set_logging()
        device = select_device(opt['device'])
        half = device.type != 'cpu'
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    # a
    # weights, imgsz = opt['weights'], opt['img-size']
    # device = select_device(opt['device'])
    # model = attempt_load(weights, map_location=device)
    # checkpoint = torch.load(fp_weight)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()
    
    
    main(fp_img, model, opt, model_size, crop_sizes, fp_out_shape)
    total_minutes = (time.time() - x)/60
    # print(total_minutes)
    minutes = int(total_minutes)
    decimal_part = total_minutes - minutes
    # print(decimal_part)
    seconds = int(decimal_part * 60)
    
    with open(fp_out_shape.replace('.shp', '.txt'), 'w') as file:
    # Ghi nội dung vào tệp tin
        file.write(f'Thoi gian cua mo hinh predict la : {minutes}m{seconds}s')
    
    