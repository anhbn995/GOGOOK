import os
import sys
import time
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from mrcnn.evaluate import build_coco_results, evaluate_coco
from mrcnn.dataset import MappingChallengeDataset

import zipfile
import urllib.request
import shutil

ROOT_DIR = os.getcwd()
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config

# import imgaug as ia
# from imgaug import augmenters as iaa
import argparse
# ia.seed(1)

# from mrcnn import model as modellib, utils
from mrcnn import model as modellib, utils

trans = 67
vals = 20
# trans = 1300
# vals = 300

def main(pretrain_model_path=None, train_step=trans,val_step=vals,roi_positive_ratio=0.33,rpn_nms_th=0.3,max_gt_instances=30):
    data_dir = r"/home/skm/SKM/nam/codeee/train_data"
    log_dir = r"/home/skm/SKM/nam/codeee/"
    model_name = "ship"

    model_size=512
    stage_1=30
    stage_2=50
    stage_3=500

    class MappingChallengeConfig(Config):
        """Configuration for training on data in MS COCO format.
        Derives from the base Config class and overrides values specific
        to the COCO dataset.
        """
        # Give the configuration a recognizable name
        NAME = model_name
        IMAGES_PER_GPU = 2

        # Uncomment to train on 8 GPUs (default is 1)
        GPU_COUNT = 1

        # Number of classes (including background)
        NUM_CLASSES = 1 + 1  # 1 Backgroun + 1 Building
        IMAGE_CHANNEL_COUNT = 3
        STEPS_PER_EPOCH = train_step
        VALIDATION_STEPS = val_step
        RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
        # RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
        MASK_SHAPE = [28, 28]
        BACKBONE = "resnet101"
        USE_MINI_MASK = True
        MINI_MASK_SHAPE = (56, 56)
        IMAGE_RESIZE_MODE = "square"
        DETECTION_MAX_INSTANCES = 300
        MAX_GT_INSTANCES = max_gt_instances
        ROI_POSITIVE_RATIO = roi_positive_ratio
        RPN_NMS_THRESHOLD = rpn_nms_th
        # RPN_NMS_THRESHOLD = 0.9
        # ROI_POSITIVE_RATIO = 0.5
        IMAGE_MAX_DIM=model_size
        IMAGE_MIN_DIM=model_size
        MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    config = MappingChallengeConfig()
    config.display()

    model = modellib.MaskRCNN(mode="training", config=config, model_dir=log_dir)
    
#     Load pretrained weights
    if pretrain_model_path:
        model_path = pretrain_model_path
        # model.load_weights(model_path, by_name=True)
    else:
        model_path=model.get_imagenet_weights()
    model.load_weights(model_path, by_name=True)#, exclude='conv1'
    
#     Load training dataset
    dataset_train = MappingChallengeDataset()
    dataset_train.load_dataset(dataset_dir=os.path.join(data_dir, "train"), load_small=False)
    dataset_train.prepare()
#     for i in dataset_train:
#         print('-------------------------')
#     Load validation dataset
    dataset_val = MappingChallengeDataset()
    val_coco = dataset_val.load_dataset(dataset_dir=os.path.join(data_dir, "val"), load_small=False, return_coco=True)
    dataset_val.prepare()
    if stage_1:
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=stage_1,
                    layers='heads',
                    augmentation=None)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    if stage_2:
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=stage_2,
                    layers='4+',
                    augmentation=None)

    # Training - Stage 3
    # Fine tune all layers
    
    if stage_3:
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE/10,
                    epochs=stage_3,
                    layers='all',
                    augmentation=None)

if __name__ == "__main__":
    # pretrain_model_path = r"/mnt/Nam/tmp_Nam/tree_counting/weights/test20220215T1656/mask_rcnn_test_0095.h5"
    main()
