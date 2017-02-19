import os
import pandas as pd
from bs4 import BeautifulSoup
import voc_utils
from more_itertools import unique_everseen

root_dir = '/media/ws/000F9A5700006688/TDDOWNLOAD/SUN2012pascalformat/SUN2012pascalformat/'
img_dir = os.path.join(root_dir, 'JPEGImages')
ann_dir = os.path.join(root_dir, 'Annotations')
set_dir = os.path.join(root_dir, 'ImageSets', 'Main')

#all_files = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'trainval', 'tvmonitor', 'val']
#image_sets = sorted(list(set([filename.replace('.txt', '').strip().split('_')[0] for filename in all_files])))
# print image_sets

xml = ''
path_sun = ann_dir+'/sun_aaaatuxrpwrbvtuv.xml'
with open(path_sun) as f:
    xml = f.readlines()
print xml
xml = ''.join([line.strip("\t") for line in xml])
#print xml
a = BeautifulSoup(xml)
print a




