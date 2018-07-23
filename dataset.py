
import os
import cv2
import random
from collections import namedtuple

import torch.utils.data as data
import numpy as np

import utils.plot_tools as plot_tools
import utils.data_augmentation as aug
from conf import settings


#only support voc dataset
class YOLODataset_Train(data.Dataset):

    def __init__(self, voc_root):
        print('data initializing.......')

        Box = namedtuple('Box', 'cls_id, x y w h')

        #variable to store boxes 
        self.labels = []

        #variable to store images path
        self.images_path = []
        with open('data/train_voc.txt') as train_file:
            for line in train_file.readlines():
                # add image path
                self.images_path.append(line.strip())

                image_id = os.path.basename(line.strip()).split('.')[0]
                with open(os.path.join(settings.LABLE_PATH, image_id + '.txt')) as label_file:
                    #get boxes per image
                    boxes = []
                    for box in label_file.readlines():
                        paras = [float(p) for p in box.strip().split()]
                        paras[0] = int(paras[0]) #change cls_id to int
                        box = Box(*paras)

                        boxes.append(box)

                    self.labels.append(boxes)    
                
    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index])
        boxes = self.labels[index]

        #data augment
        image, boxes = aug.random_horizontal_flip(image, boxes)
        image = aug.random_bright(image)
        image = aug.random_hue(image)
        image = aug.random_saturation(image)
        image = aug.random_gaussian_blur(image)
        image, boxes = aug.random_affine(image, boxes)
        image, boxes = aug.random_crop(image, boxes)

        plot_tools.plot_image_bbox(image, boxes)
        

    def __len__(self):
        return len(self.images_path) 


yolo_data = YOLODataset_Train(settings.ROOT_VOC_PATH)

import cProfile

cProfile.runctx('yolo_data[3]', globals(), None)


for i in range(10):
    yolo_data[random.randint(1, len(yolo_data))]



