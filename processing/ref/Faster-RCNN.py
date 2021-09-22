import os
import numpy as np
from object_detection.pascal_parser import parser
from object_detection.data_generator import gt_generator
from object_detection.config_setting import config

# data load
config_set = config
pascal_dir = "E:\\Deep_learning_dataset\\Pascal_2012\\VOCtrainval_11-May-2012\\VOCdevkit"
all_imgs, class_count, class_mapping = parser(pascal_dir=pascal_dir)
gt_generator(all_imgs_info = all_imgs,
             set = config_set)