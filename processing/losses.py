import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

def rpn_loss_cls(num_anchors):

    def rpn_loss_cls_fixed_num(y_true, y_pred):
        return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])

    return rpn_loss_cls_fixed_num

def rpn_loss_regr(num_anchors):

    def rpn_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, :, 4 * num_anchors:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)
        r = lambda_rpn_regr * K.sum(
            y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

        return lambda_rpn_regr * K.sum(
            y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

    return rpn_loss_regr_fixed_num

# def converter(arg):
#   arg = tf.convert_to_tensor(arg, dtype=tf.float32)
#   return arg
#
# def rpn_cls_loss(gt_cls, pred_cls):
#     """
#
#     :param gt_cls: shape = [num_anchor*height*width, 1]
#     :param pred_cls: shape = [height, width, 2*num_anchor]
#     :return:
#     """
#     gt_cls_label = np.zeros([len(gt_cls), 2])
#     gt_cls_label[gt_cls==1,1]=1
#     gt_cls_label[gt_cls==0,0]=1
#
#     gt_cls_label = converter(gt_cls_label)
#     pred_cls = tf.reshape(pred_cls,
#                           shape=[-1, 2])
#     cls_loss = tf.reduce_sum(K.log(pred_cls)*gt_cls_label)
#
# def rpn_reg_loss(gt_reg, gt_cls, pred_reg):
#     """
#
#     :param gt_reg: shape = [num_anchor*height*width, 4]
#     :param pred_reg: shape = [height, width, 4*num_anchor]
#     :return:
#     """
#
#     valid_index = gt_cls != -1
#     gt_reg = converter(gt_reg)
#     pred_reg = tf.reshape(pred_reg,
#                           shape=[-1, 4])
#     reg_loss = tf.reduce_sum(tf.math.square(gt_reg-pred_reg)[valid_index])





