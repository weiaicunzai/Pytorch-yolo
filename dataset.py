
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

        #Box definition
        Box = namedtuple('Box', 'cls_id x y w h')

        #variable to store boxes 
        self.labels = []

        #variable to store images path
        self.images_path = []

        self.S = settings.S
        self.B = settings.B
        self.classes = settings.CLASSES
        self.img_size = settings.IMG_SIZE
        self.label_path = settings.LABLE_PATH
        self.cell_size = int(self.img_size / self.S)

        with open('data/train_voc.txt') as train_file:
            for line in train_file.readlines():

                # add image path
                self.images_path.append(line.strip())
                image_id = os.path.basename(line.strip()).split('.')[0]
                with open(os.path.join(self.label_path, image_id + '.txt')) as label_file:

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

        #"""For data augmentation we introduce random scaling and
        #translations of up to 20% of the original image size. We
        #also randomly adjust the exposure and saturation of the im-
        #age by up to a factor of 1.5 in the HSV color space."""

        #data augment(I decide to use more augmentation than original paper)
        image = aug.random_bright(image)
        image = aug.random_hue(image)
        image = aug.random_saturation(image)
        image = aug.random_gaussian_blur(image)
        image, boxes = aug.random_horizontal_flip(image, boxes)
        image, boxes = aug.random_affine(image, boxes)
        image, boxes = aug.random_crop(image, boxes)
        image, boxes = aug.resize(image, boxes, (self.img_size, self.img_size))
            
        #rescale to 0 - 1
        image = image / float(255)
        target = self._encode(image, boxes)

        #plot_tools.plot_compare(image, target, boxes)
        return image, target
        #plot_tools.plot_image_bbox(image, boxes)
        

    def __len__(self):
        return len(self.images_path) 


    def _encode(self, image, boxes):
        """Transform image and boexs to a (7 * 7 * 30)
        numpy array(bbox + confidence + bbox + confidence
        + class_num)

        Args:
            image: numpy array, read by opencv
            boxes: namedtuple object
        
        Returns:
            a 7*7*30 numpy array
        """

        target = np.zeros((self.S, self.S, self.B * 5 + len(self.classes)))
        for box in boxes:
            cls_id, x, y, w, h = plot_tools.unnormalize_box_params(box, image.shape)
            col_index = int(x / self.cell_size)
            row_index = int(y / self.cell_size)

            # assign confidence score

            #"""Formally we define confidence as Pr(Object) ∗ IOU truth
            #pred . If no object exists in that cell, the confidence 
            #scores should be zero. Otherwise we want the confidence score 
            #to equal the intersection over union (IOU) between the 
            #predicted box and the ground truth."""
            target[row_index, col_index, 4] = 1
            target[row_index, col_index, 9] = 1

            #assign class probs

            #"""Each grid cell also predicts C conditional class proba-
            #bilities, Pr(Class i |Object)."""
            target[row_index, col_index, 10 + cls_id] = 1

            #assign x,y,w,h
            target[row_index, col_index, :4] = box.x, box.y, box.w, box.h
            target[row_index, col_index, 5:9] = box.x, box.y, box.w, box.h


            
        return target




yolo_data = YOLODataset_Train(settings.ROOT_VOC_PATH)

import cProfile

cProfile.runctx('yolo_data[3]', globals(), None)


for i in range(14):
    yolo_data[random.randint(1, len(yolo_data))]



