"""global settings for yolo, don't use this
module directly, use conf.settings instead

The purpose of not using this module directly
is that the variable were imported into different
modules can't share their updates, if we dont
add variable namespace explicitly which is their
modeul name 'global_setings'

Author: baiyu
"""

import os


# root path to voc dataset
ROOT_VOC_PATH = "/media/baiyu/A2FEE355FEE31FF1/VOCdevkit (2)/VOC2012"
ANNO_PATH = os.path.join(ROOT_VOC_PATH, 'Annotations')
IMAGE_PATH = os.path.join(ROOT_VOC_PATH, 'JPEGImages')
LABLE_PATH = os.path.join(ROOT_VOC_PATH, 'labels')

# voc dataset classes
CLASSES = ["aeroplane", "bicycle", "bird", "boat", 
           "bottle", "bus", "car", "cat", "chair", 
           "cow", "diningtable", "dog", "horse", 
           "motorbike", "person", "pottedplant", "sheep", 
           "sofa", "train", "tvmonitor"]

# num of voc test dataset
NUM_OF_TEST = 5000

# augment probablity
# probablity to use augmentation
AUG_PROB = 0.5

# a float number below 1.0 the brightness will be in 
# range brightness * [1.0 - brightness, 1.0 + brightness]
BRIGHTNESS = 0.7

# a float number below 1.0 the factor to change the 
# hue of an image hue = hue * [1.0 - hue_factor, 1. + hue_factor]
HUE_FACTOR = 0.7

# a float number below 1.0 the factor to change the
# staturation of an image
STATURATION_FACTOR = 0.7

# affine scale factor
# transformed image at least are 80% of 
# the original width and height
AFFINE_SCALE_FACTOR = 0.9

# affine shift facotor
# transformed image at most shifted
# 20% of the original width and height
AFFINE_SHIFT_FACTOR = 0.2

# crop_resize_ratio
# resize image to at most (1 + ratio) times
CROP_JITTER = 0.2


# """For evaluating YOLO on PASCAL VOC, we use S = 7,
# B = 2. PASCAL VOC has 20 labelled classes so C = 20.
# Our final prediction is a 7 × 7 × 30 tensor."""

# grid number
S = 7

# bounding box number
B = 2

# lambda coord
LAMBDA_COORD = 5

# lambda noobj
LAMBDA_NOOBJ = .5

# image size
IMG_SIZE = 448

