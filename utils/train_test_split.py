"""This module split dataset into
2 sub datasets, train and test

Author: baiyu
"""

import os
import random

from conf import settings



image_path = settings.IMAGE_PATH
label_path = settings.LABLE_PATH
labels = os.listdir(label_path)
image_names = [label.replace('.txt', '.jpg') for label in labels]

random.shuffle(image_names)

#training_dataset
with open(os.path.join('data' ,'train_voc.txt'), 'w') as train_file:
    for image_name in image_names[settings.NUM_OF_TEST:]:
        train_file.write(os.path.join(image_path, image_name) + '\n')      

#test_dataset
with open(os.path.join('data', 'test_voc.txt'), 'w') as test_file:
    for image_name in image_names[:settings.NUM_OF_TEST]:
        test_file.write(os.path.join(image_path, image_name) + '\n')