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

#num of voc test dataset
NUM_OF_TEST = 5000







