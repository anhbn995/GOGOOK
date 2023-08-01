# import geopandas
import rasterio.mask
import rasterio
from rasterio import windows
# import geopandas as gp
from itertools import product
import numpy as np
import glob, os
import sys
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
from shapely.geometry import Polygon, box, mapping, MultiPolygon, Point
from shapely.strtree import STRtree
import fiona
import numpy as np
import glob, os
from multiprocessing.pool import Pool
from itertools import product
import math
import random
import tensorflow as tf
import shutil
import time
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, ZeroPadding2D, BatchNormalization, MaxPool2D
from tensorflow.keras.regularizers import l2

strides = np.array([8, 16, 32])
batch_sizes = 4
input_sizes = 608
output_sizes = np.array(input_sizes) // strides
yolo_anchor = [[[36.32, 75.11], [72.11, 56.51], [11.37, 68.45]],
               [[78.14, 75.84], [86.23, 25.87], [85.78, 66.22]],
               [[94.22, 65.09], [103.87, 86.43], [129.30, 104.47]]]
anchors = (np.array(yolo_anchor).T/ strides).T
num_anchor = anchors.shape[1]

class BatchNormalization(BatchNormalization):
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type = "leaky"):
    if downsample:
        input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
                  padding=padding, use_bias=not bn, kernel_regularizer=l2(0.0005),
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                  bias_initializer=tf.constant_initializer(0.))(input_layer)
    if bn:
        conv = BatchNormalization()(conv)
    if activate == True:
        if activate_type == "leaky":
            conv = LeakyReLU(alpha=0.1)(conv)
        elif activate_type == "mish":
            conv = mish(conv)     
    return conv

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')

def residual_block(input_layer, filter_num, num_blocks):
    input_layer = convolutional(input_layer, (3, filter_num), downsample=True)
    for i in range(num_blocks):
        short_cut = input_layer
        conv = convolutional(input_layer, filters_shape=(1, filter_num//2))
        conv = convolutional(conv       , filters_shape=(3, filter_num))
        input_layer = short_cut + conv
    return input_layer

def darknet53(input_data):
    input_data = convolutional(input_data, (3, 32))
    input_data = residual_block(input_data, 64, 1)
    input_data = residual_block(input_data, 128, 2)
    input_data = residual_block(input_data, 256, 8)
    route_1 = input_data
    input_data = residual_block(input_data, 512, 8)
    route_2 = input_data
    input_data = residual_block(input_data, 1024, 4)
    return route_1, route_2, input_data

def cspresidual_block(input_layer, filter_num, num_blocks, check = True):
    input_layer = convolutional(input_layer, (3, filter_num), downsample=True, activate_type="mish")
    route = input_layer
    if check:
        num_filter = filter_num//2
    else:
        num_filter = filter_num
    
    route = convolutional(route, (1, num_filter), activate_type="mish")
    input_layer = convolutional(input_layer, (1, num_filter), activate_type="mish")
    for i in range(num_blocks):
        conv = convolutional(input_layer, filters_shape=(1, filter_num//2), activate_type="mish")
        conv = convolutional(conv       , filters_shape=(3, num_filter), activate_type="mish")
        input_layer = input_layer + conv
        
    input_layer = convolutional(input_layer, (1, num_filter), activate_type="mish")
    input_layer = tf.concat([input_layer, route], axis=-1)
    input_layer = convolutional(input_layer, (1, filter_num), activate_type="mish")
    
    return input_layer

def cspdarknet53(input_data):
    input_data = convolutional(input_data, (3,  32), activate_type="mish")
    input_data = cspresidual_block(input_data, 64, 1, check = False)
    input_data = cspresidual_block(input_data, 128, 2)
    input_data = cspresidual_block(input_data, 256, 8)
    route_1 = input_data
    input_data = cspresidual_block(input_data, 512, 8)
    route_2 = input_data
    input_data = cspresidual_block(input_data, 1024, 4)
    
    input_data = convolutional(input_data, (1, 512))
    input_data = convolutional(input_data, (3, 1024))
    input_data = convolutional(input_data, (1, 512))

    input_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='SAME', strides=1), tf.nn.max_pool(input_data, ksize=9, padding='SAME', strides=1), tf.nn.max_pool(input_data, ksize=5, padding='SAME', strides=1), input_data], axis=-1)
    input_data = convolutional(input_data, (1, 512))
    input_data = convolutional(input_data, (3, 1024))
    input_data = convolutional(input_data, (1, 512))

    return route_1, route_2, input_data

def make_last_layers(conv, num_filters, num_classes):
    conv = convolutional(conv, (1, 1, num_filters))
    conv = convolutional(conv, (3, 3, num_filters*2))
    conv = convolutional(conv, (1, 1, num_filters))
    conv = convolutional(conv, (3, 3, num_filters*2))
    conv = convolutional(conv, (1, 1, num_filters))
    conv_branch = convolutional(conv, (3, 3, num_filters*2))
    conv_box = convolutional(conv_branch, (1, 1, num_anchor*(num_classes+5)), activate=False, bn=False)
    return conv, conv_box

def YOLOv3(input_layer, num_classes):
    route_1, route_2, conv = darknet53(input_layer)

    conv, y1 = make_last_layers(conv, 512, num_classes)
    conv = convolutional(conv, (1, 256))
    conv = upsample(conv)
    conv = tf.concat([conv, route_2], axis = -1)

    conv, y2 = make_last_layers(conv, 256, num_classes)
    conv = convolutional(conv, (1, 128))
    conv = upsample(conv)
    conv = tf.concat([conv, route_1], axis = -1)

    conv, y3 = make_last_layers(conv, 128, num_classes)
    return [y3, y2, y1]

def make_yolov4(conv, route, num_filters, num_classes, check = False):
    if check:
        route_x = conv
    conv = convolutional(conv, (1, num_filters))  
    if check:
        conv_bbox = convolutional(conv, (1, num_anchor * (num_classes + 5)), activate=False, bn=False)
        conv = convolutional(route_x, (3, num_filters), downsample=True)
        conv = tf.concat([conv, route], axis=-1) 
    else:
        conv_bbox = upsample(conv)    
        route = convolutional(route, (1, num_filters))
        conv = tf.concat([route, conv_bbox], axis=-1) 
        
    conv = convolutional(conv, (1, num_filters))
    conv = convolutional(conv, (3, num_filters*2))
    conv = convolutional(conv, (1, num_filters))
    conv = convolutional(conv, (3, num_filters*2))
    conv = convolutional(conv, (1, num_filters))
    
    return conv, conv_bbox

def YOLOv4(input_layer, num_classes):
    route_1, route_2, conv = cspdarknet53(input_layer)

    route_n = conv
    conv, _ = make_yolov4(conv, route_2, 256, num_classes)
    route_2 = conv
    conv, _ = make_yolov4(conv, route_1, 128, num_classes)
    conv, conv_sbbox = make_yolov4(conv, route_2, 256, num_classes, check=True)
    conv, conv_mbbox = make_yolov4(conv, route_n, 512, num_classes, check=True)
    
    conv = convolutional(conv, (3, 1024))
    conv_lbbox = convolutional(conv, (1, num_anchor * (num_classes + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def decode(conv_output, NUM_CLASS, i=0):
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, num_anchor, 5 + NUM_CLASS))
    conv_raw_dxdy = conv_output[..., :2]
    conv_raw_dwdh = conv_output[..., 2:4]
    conv_raw_conf = conv_output[..., 4:5]
    conv_raw_prob = conv_output[..., 5: ]

    y = tf.range(output_size, dtype=tf.int32)
    y = tf.expand_dims(y, -1)
    y = tf.tile(y, [1, output_size])
    x = tf.range(output_size,dtype=tf.int32)
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, num_anchor, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * strides[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * anchors[i]) * strides[i]

    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def Create_Yolo(input_size=608, channels=3, training=False):
    num_classes = 1
    input_layer  = Input([input_size, input_size, channels])
    conv_tensors = YOLOv3(input_layer, num_classes)
    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, num_classes, i)
        if training: 
            output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    Yolo = tf.keras.Model(input_layer, output_tensors)
    return Yolo

def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    # inter_section = tf.clip_by_value(right_down - left_up, clip_value_min= 0.0, clip_value_max=1e+5)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = 1.0 * inter_area / union_area

    iou = tf.where(tf.math.is_nan(iou), 0.0, iou)
    return iou

def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    # inter_section = tf.clip_by_value(right_down - left_up, clip_value_min= 0.0, clip_value_max=1e+5)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area
    iou = tf.where(tf.math.is_nan(iou), 0.0, iou)

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])

    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    # enclose = tf.clip_by_value(enclose_right_down - enclose_left_up, clip_value_min= 0.0, clip_value_max=1e+5)
    enclose_area = enclose[..., 0] * enclose[..., 1] 
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area
    giou = tf.where(tf.math.is_nan(giou), -1.0, giou)

    return giou

def compute_loss(pred, conv, label, bboxes, i=0):
    NUM_CLASS = 1
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = strides[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, num_anchor, 5 + NUM_CLASS))

    conv_raw_conf = conv[..., 4:5]
    conv_raw_prob = conv[..., 5:]

    pred_xywh     = pred[..., 0:4]
    pred_conf     = pred[..., 4:5]

    label_xywh    = label[..., 0:4]
    respond_bbox  = label[..., 4:5]
    label_prob    = label[..., 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[...,2:3] * label_xywh[...,3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < 0.7, tf.float32 ) 

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss

def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)

    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [input_size , input_size])], axis=-1)
    invalid_mask_1 = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    invalid_mask_2 = np.logical_or(np.any(pred_coor[:, :2] == 0, axis= -1), np.any(pred_coor[:, 2:] == input_size, axis= -1))
    invalid_mask = np.logical_or(invalid_mask_1, invalid_mask_2)
    pred_coor[invalid_mask] = 0

    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    scores = pred_conf * pred_prob.T
    score_mask = scores[0] > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores = pred_coor[mask], scores[0][mask]

    return np.concatenate([coors, scores[:, np.newaxis]], axis=-1)

def nms(cls_bboxes, iou_threshold):
    best_bboxes = []
    while len(cls_bboxes) > 0:
        max_ind = np.argmax(cls_bboxes[:, 4])
        best_bbox = cls_bboxes[max_ind]
        best_bboxes.append(best_bbox)
        cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
        iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
        weight = np.ones((len(iou),), dtype=np.float32)

        iou_mask = iou > iou_threshold
        weight[iou_mask] = 0.0

        cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
        score_mask = cls_bboxes[:, 4] > 0.
        cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def get_tiles(ds, width, height, strides = 500):
    nols, nrows = ds.meta['width'], ds.meta['height']
    print(nols, nrows)
    offsets = product(range(0, nols, strides), range(0, nrows, strides))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    offset = []
    for col_off, row_off in offsets:
        if row_off + width > nrows:
            row_off = nrows - width
        if  col_off + height > nols:
            col_off = nols - height
        offset.append((col_off, row_off))
    offset = set(offset)
    for col_off, row_off in offset: 
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform
        

def detect_image(Yolo, image_path, input_size=608, score_threshold=0.15, iou_threshold=0.2):
    image_data = image_path[np.newaxis, ...].astype(np.float32)
    # print(np.unique(image_data),'nam')
    pred_bbox = Yolo.predict(image_data)
    # print(np.unique(pred_bbox[1]), 'bo')
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    
    bboxes = postprocess_boxes(pred_bbox, image_path, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold)
    return bboxes


def check(x,y):
    for k, i in enumerate(x[:,1]):
        if (i.intersection(y[1])).area > 5.6259513540630335e-12:
            x[:,0][k]=0
    return x[x[:,0]>0]

def checks(data1, data2):
    for i in range(len(data1)):
        if np.all((data1[i]) != data2[i]):
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        else:
            print('n')


if __name__ == "__main__":
    yolo1 = Create_Yolo(input_size=608)
    # yolo2 = Create_Yolo(input_size=608)

    yolo1.load_weights("yolov3_tree")
    # print(yolo.trainable_weights)
    # yolo.load_weights("yolov3_tree")
    gg = yolo1.trainable_weights
    image_list = 'Orthomosaic.tif'
        
    inds = rasterio.open(image_list)
    # img_data = inds.read()[:3].transpose(1,2,0)
    tile_width, tile_height = 608, 608
    dicts = {}
    test = []
    # yolo.load_weights("yolov3_tree")
    for window, transform in get_tiles(inds, tile_width, tile_height):
        data = inds.read(window=window)[:3].transpose(1,2,0)
        # print(data.mean())
        # data = img_data[window.row_off: window.row_off+ window.height, window.col_off: window.col_off+window.width, :]
        # yolo2.load_weights("yolov3_tree")
        boxx = detect_image(yolo1, data, input_size=608, score_threshold=0.15, iou_threshold= 0.4)
        checks(gg,yolo1.trainable_weights)
        print(len(boxx))
    #     for i in boxx:
    #         x_min = i[0] * transform[0] + transform[2]
    #         x_max = i[2] * transform[0] + transform[2]
    #         y_max = i[1] * transform[4] + transform[5]
    #         y_min = i[3] * transform[4] + transform[5]
    #         center = box(x_min, y_min, x_max, y_max)
    #         dicts.update({tuple(np.array(center.centroid)): (i[4], center)})
    #         test.append(center.centroid)

    with rasterio.open(image_list) as src:
        transform = src.transform
        projstr = src.crs.to_string()
        check_epsg = src.crs.is_epsg_code
        if check_epsg:
            epsg_code = src.crs.to_epsg()
        else:
            epsg_code = None
    if epsg_code:
        out_crs = {'init':'epsg:{}'.format(epsg_code)}
    else:
        out_crs = projstr

    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int'},
    }

    with fiona.open('test4.shp', 'w', crs=out_crs, driver='ESRI Shapefile', schema=schema) as c:
        with rasterio.open(image_list) as inds:
            tile_width, tile_height = 2000, 2000
            meta = inds.meta.copy()
            super_data = []
            best_bboxes = []
            ids = 0
            for window, transform in get_tiles(inds, tile_width, tile_height, 2000):
    #             boxx = box(*transform * (0, 0),*transform * (tile_width, tile_height))
    #             boxx = box(*transform * (0, 0),*(np.array(transform * (tile_width, tile_height))-0.4))
                boxx = box(*(np.array(transform * (0, 0))),*transform * (tile_width, tile_height))
                cls_bboxes = np.array([dicts[tuple(np.array(i))] for i in STRtree(test).query(boxx)])
                while len(cls_bboxes) > 0:
                    max_ind = np.argmax(cls_bboxes[:, 0])
                    best_bbox = cls_bboxes[max_ind]
                    c.write({
                        'geometry': mapping(best_bbox[1]),
                        'properties': {'id': ids}
                    })
                    cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                    cls_bboxes = check(cls_bboxes, best_bbox)
                    ids+=1
                    print(ids)