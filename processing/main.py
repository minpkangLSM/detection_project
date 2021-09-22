"""
Faster R-CNN implementation from 21.07.29.
Kangmin Park, LSM, The Univ. of Seoul.
"""
import sys
sys.path.append("..\\obj_detection")
from utils.pascal_parser import parsing_pascal
from utils.data_generator import *
from utils import config
from losses import rpn_loss_cls, rpn_loss_regr
from backbones import resnet
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from utils import roi_helpers

# config
config = config.config()

## Data preparation
pascal_dir = "E:\\Deep_learning_dataset\\Pascal_2012\\VOCtrainval_11-May-2012\\VOCdevkit"
img_data, classes_count, classes_mapping = parsing_pascal(pascal_dir = pascal_dir)

if "background" not in classes_count :
    classes_count["background"] = 0
    classes_mapping["background"] = len(classes_mapping)

train_data = [img for img in img_data if img["imageset"] == "train"]
valid_data = [img for img in img_data if img["imageset"] == "val"]

## Data generation
train_gen = get_gt_featuremap(all_img_data=train_data, config=config)

## RPN
input_shape_img = (None, None, 3)
img_input = keras.Input(shape=input_shape_img)
num_anchors = len(config.anchor_box_ratios)*len(config.anchor_box_scales)

shared_layers = resnet.ResNet50(input_tensor=img_input)

rpn_pred_regr = resnet.RPN(output_feature=shared_layers,
                           num_anchors=num_anchors)
model_rpn = Model(inputs=img_input,
                  outputs=rpn_pred_regr[:2])
model_rpn.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors)])

num_epochs = 100

for epoch_num in range(num_epochs):

    while True :

        X, Y, img_data = next(train_gen)
        loss_rpn = model_rpn.train_on_batch(X,Y)
        p_rpn = model_rpn.predict_on_batch(X) # rpn_cls, rpn_regr
        # p_rpn[0] : rpn cls feature / p_rpn[1] : rpn regr feature
        R = roi_helpers.rpn_to_roi(p_rpn[0], p_rpn[1], config,
                                   use_regr=True, overlap_threshold=0.7, max_boxes=300)
        X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, config, classes_mapping)







