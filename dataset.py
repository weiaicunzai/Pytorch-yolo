
import os
import cv2
from collections import namedtuple

import torch.utils.data as data

from conf import settings


#classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

#only support voc dataset
class YOLODataset_Train(data.Dataset):

    def __init__(self, voc_root):
        print('data initializing.......')

        Box = namedtuple('Box', 'cls_id, x y w h')
        labels = []
        images_path = []
        with open('data/train_voc.txt') as train_file:
            for line in train_file.readlines():
                images_path.append(line.strip())
                print(line.strip())


                image_id = os.path.basename(line.strip()).split('.')[0]
                with open(os.path.join(settings.LABLE_PATH, image_id + '.txt')) as label_file:
                    boxes = []
                    for box in label_file.readlines():
                        paras = [float(p) for p in box.strip().split()]
                        paras[0] = int(paras[0]) #change cls_id to int
                        box = Box(*paras)

                        boxes.append(box)
                        

        #sorted(self.img)
        #sorted(self.labels)
        #for i in range(len(self.img)):
        #    print(self.img[i], self.labels[i])
        #for i in range(10):
        #    self._plot_image_bbox(os.path.join(self.image_path, self.img[i * 349 - 25]))
        #print(label_name)

    
    #def _plot_image_bbox(self, image_path):
    #    image_name = os.path.basename(image_path)
    #    image = cv2.imread(image_path)

    #    label_name = os.path.splitext(image_name)[0] + '.txt'
    #    labels = open(os.path.join(self.labels_path, label_name), 'r')
    #    boxes = []
    #    for line in labels.readlines():
    #        box = [float(x) for x in line.strip().split()]
    #        boxes.append(box)

    #    for box in boxes:
    #        shape = image.shape
    #        hight = shape[0]
    #        width = shape[1]

    #        cls_index, x, y, w, h = box

    #        x *= width
    #        w *= width
    #        y *= hight
    #        h *= hight

    #        #draw bbox
    #        top_left = (int(x - w / 2), int(y - h / 2))
    #        bottom_right = (int(x + w / 2), int(y + h / 2))
    #        cv2.rectangle(image, top_left, bottom_right, (255, 220, 33))

    #        #draw Text background rectangle
    #        text_size, baseline = cv2.getTextSize(classes[int(cls_index)], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    #        cv2.rectangle(image, (top_left[0], top_left[1] - text_size[1]),
    #                             (top_left[0] + text_size[0], top_left[1]),
    #                             (255, 220, 33),
    #                             -1)
    #        
    #        #draw text
    #        cv2.putText(image, classes[int(cls_index)], 
    #                           top_left,
    #                           cv2.FONT_HERSHEY_DUPLEX,
    #                           0.4,
    #                           (255, 255, 255),
    #                           1,
    #                           8,
    #                           )
    #    cv2.imshow('test', image)
    #    cv2.waitKey(0)


YOLODataset_Train(settings.ROOT_VOC_PATH)