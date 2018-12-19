# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf

# ------------------------------------------------
NET_NAME = 'resnet50'#'resnet100'  # 'mobilenetv2' 'resnet50'
ADD_BOX_IN_TENSORBOARD = True
#------------------------------------------ convert data to tfrecofr config
BIN_DATA = 0 # whether read image data from binary

# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "3"
SHOW_TRAIN_INFO_INTE = 5000
SMRY_ITER = 100000

# ------------------------------------------ Train config
BN_USE = True 
RESTORE_FROM_RPN = False
IS_FILTER_OUTSIDE_BOXES = True
FIXED_BLOCKS = 1  # allow 0~3

M1_LOCATION_LOSS_WEIGHT = 1.
M1_CLASSIFICATION_LOSS_WEIGHT = 2.0

M2_LOCATION_LOSS_WEIGHT = 1.0
M2_CLASSIFICATION_LOSS_WEIGHT = 2.0

M3_LOCATION_LOSS_WEIGHT = 1.0
M3_CLASSIFICATION_LOSS_WEIGHT = 2.0

MUTILPY_BIAS_GRADIENT = None   # 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = None   # 10.0  if None, will not clip

EPSILON = 1e-5
MOMENTUM = 0.9
LR = [0.01,0.001,0.0001,0.00001]
DECAY_STEP = [150000, 300000,450000]
# -------------------------------------------- Data_preprocess_config 
DATASET_NAME = 'WiderFace'  # 'ship', 'spacenet', 'pascal', 'coco'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
PIXEL_NORM = 128.0
IMG_LIMITATE = 0
IMG_SHORT_SIDE_LEN = 480
IMG_MAX_LENGTH = 640
CLASS_NUM = 1

# --------------------------------------------- Network_config
BATCH_SIZE = 1
INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01)
BBOX_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.001)
WEIGHT_DECAY = 0.00005 if NET_NAME.startswith('Mobilenet') else 0.0001
M1_CHANNELS = 128
M2_CHANNELS = 256
M3_CHANNELS = 256

# ---------------------------------------------Anchor config
BASE_ANCHOR_SIZE_LIST = [16]
ANCHOR_STRIDE = [16, 16, 32]
M1_ANCHOR_SCALES = [0.5, 1.0, 1.5]
M2_ANCHOR_SCALES = [2.0, 4.0, 8.0]
M3_ANCHOR_SCALES = [8.0,16.0, 32.0]
ANCHOR_RATIOS = [1.]
ROI_SCALE_FACTORS = [10., 10., 5.0, 5.0]
ANCHOR_SCALE_FACTORS = None

EXTRA_SHIFTED_ANCHOR = False
FACE_SHIFT_JITTER = False
NUM_SHIFT_JITTER = 4

# -------------------------------------------detection module config
M1_NMS_IOU_THRESHOLD = 0.1
M1_SIGMA = 1.0

M2_NMS_IOU_THRESHOLD = 0.2
M2_SIGMA = 1.0

M3_NMS_IOU_THRESHOLD = 0.3
M3_SIGMA = 1.0

TOP_K_NMS_TRAIN = 0
MAXIMUM_PROPOSAL_TARIN = 2000
TOP_K_NMS_TEST = 0
MAXIMUM_PROPOSAL_TEST = 300

M1_IOU_POSITIVE_THRESHOLD = 0.25
M1_IOU_NEGATIVE_THRESHOLD_UP = 0.1
M1_IOU_NEGATIVE_THRESHOLD_DOWN = 0.0   # 0.1 < IOU < 0.3 is negative
M1_MINIBATCH_SIZE = 256  # if is -1, that is train with OHEM
M1_POSITIVE_RATE = 0.25

M2_IOU_POSITIVE_THRESHOLD = 0.5
M2_IOU_NEGATIVE_THRESHOLD_UP = 0.3
M2_IOU_NEGATIVE_THRESHOLD_DOWN = 0.0   # 0.1 < IOU < 0.5 is negative
M2_MINIBATCH_SIZE = 256  # if is -1, that is train with OHEM
M2_POSITIVE_RATE = 0.25

M3_IOU_POSITIVE_THRESHOLD = 0.5
M3_IOU_NEGATIVE_THRESHOLD_UP = 0.3
M3_IOU_NEGATIVE_THRESHOLD_DOWN = 0.0   # 0.1 < IOU < 0.5 is negative
M3_MINIBATCH_SIZE = 256  # if is -1, that is train with OHEM
M3_POSITIVE_RATE = 0.25

SHOW_SCORE_THRSHOLD = 0.9  # only show in tensorboard
SSH_NMS_MAX_BOXES_PER_CLASS = 300

ADD_GTBOXES_TO_TRAIN = False
FINAL_NMS_IOU_THRESHOLD = 0.3

