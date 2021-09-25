import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import AvgPool2D
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K

"""
COPYRIGHT : kangmin Park, Lab for Sensor and Modeling, Dept. of Geoinformatics, the Univ. of Seoul.

# reupdated
2020.07.03.~
This code is utils for DCNN(Deep convolution neural network) backbones, based on Keras(Tensorflow 2.x).
updated for Semantic segmentation model.

2021.01.19.
Updated Resblock_bn to dilation version related with DualAttentionNetwork.

2021.06.03. ~ 2021.06.17.
Updated with Transformer of ViT, SETR model

2021.06.17.
Updated with STANet module : PAM / contrastive loss function

2021.07.26.
Updated with SE module : Squeeze-and-Excitation Networks, 2019(2017)

CODE LIST
1) INCEPTION MODULE
2) XCEPTION(Navie / DeepLabV3+) MODULE
3) RESNET MODULE
4) DLINKNET MODULE
5) DENSENET MODULE
6) RESNET(DeeplabV3+) MODULE
7) HIERARCHICAL IMP.
8) SEMANTIC SEGMENTATION PREDICTION DEF.
9) VISION TRANSFORMER MODULE - MSA / MLP -> TRANSFORMER
10) Squeeze-and-Excitation Net : SE module
11) ViT / BiT(not yet)
"""

def scheduler(epoch,
              lr):

    initial_lr = 0.0001
    end_lr = 0.000001
    decay_step = 100
    lr = (initial_lr-end_lr)*(1-epoch/decay_step)+end_lr
    return lr

def SepConv_BN(x, filters, prefix, strides=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ Ref : https://github.com/bonlime/keras-deeplab-v3-plus.git
    SepConv with BN between depthwise & pointwise. Optionally add activation after BN
    Implements right "same" padding for even kernel sizes
    Args:
        x: input tensor
        filters: num of filters in pointwise convolution
        prefix: prefix before name
        stride: stride at depthwise conv
        kernel_size: kernel size for depthwise convolution
        rate: atrous rate for depthwise convolution
        depth_activation: flag to use activation between depthwise & poinwise convs
        epsilon: epsilon to use in BN layer
    """
    if strides==1:
        depth_padding = "same"
    else :
        kernel_size_effective = kernel_size+(kernel_size-1)*(rate-1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation(tf.nn.relu)(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(tf.nn.relu)(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation(tf.nn.relu)(x)

    return x

def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
# INCEPTION
def LRN(input, layer_name, depth_radius=5, bias=1, alpha=1, beta=0.5):
    """

    :param input:
    :param depth_radius: n
    :param bias: K
    :param alpha: alpha
    :param beta: beta
    :param layer_name:
    :return:
    """
    L = tf.nn.local_response_normalization(input, depth_radius=depth_radius,
                                           bias=bias, alpha=alpha, beta=beta, name=layer_name)
    return L

def auxiliary_classifier(input, keep_prob, verbose=True):
    """
    :param layer_name:
    :param is_training:
    :param input:
    :param keep_prob:
    :param label:
    :return:
    """
    avg_L = AvgPool2D(pool_size=(5,5), strides=(3,3), padding="valid")(input)
    con_L = Conv2D(filters=128, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(avg_L)
    if verbose : con_L = BatchNormalization(axis=-1, momentum=0.9)(con_L)
    con_L = Activation(activation="relu")(con_L)
    con_L = Flatten()(con_L)
    FC_L = Dropout(rate=keep_prob)(con_L)
    FC_L = Dense(units=1024, activation="relu")(FC_L)
    FC_L = Dropout(rate=keep_prob)(FC_L)
    FC_L = Dense(units=102, activation="softmax")(FC_L)

    return FC_L

def naive_module(input, A_size, B1_size, B2_size, C1_size, C2_size, D2_size, pad="same"):
    """
    :param input : input feature map
    :param A_size: output channels of leftmost part in the naive module - conv1x1
    :param B1_size: output channels of second left side part in the naive module - conv1x1
    :param B2_size: output channels of second left side part in the naive module - conv3x3
    :param C1_size: output channels of third left side part in the naive module - conv1x1
    :param C2_size: output channels of third left side part in the naive module - conv5x5
    :param D2_size: output channels of rightmost side part in the naive module - conv1x1
    :return: concat
    """
    A1 = Conv2D(filters=A_size, kernel_size=1, strides=(1,1), padding=pad, activation="relu")(input)
    B1 = Conv2D(filters=B1_size, kernel_size=1, strides=(1,1), padding=pad, activation="relu")(input)
    B2 = Conv2D(filters=B2_size, kernel_size=3, strides=(1,1), padding=pad, activation="relu")(B1)
    C1 = Conv2D(filters=C1_size, kernel_size=1, strides=(1,1), padding=pad, activation="relu")(input)
    C2 = Conv2D(filters=C2_size, kernel_size=5, strides=(1,1), padding=pad, activation="relu")(C1)
    D1 = MaxPool2D(pool_size=(3,3), strides=(1,1), padding=pad)(input)
    D2 = Conv2D(filters=D2_size, kernel_size=1, strides=(1,1), padding=pad, activation="relu")(D1)
    concat = Concatenate()([A1, B2, C2, D2])
    return concat

def A_module(input, A_size, B1_size, B2_size, C1_size, C2_size, C3_size, D2_size, pad="same"):
    """
    A module for Inception V2~3
    :param input:
    :param A_size:
    :param B1_size:
    :param B2_size:
    :param C1_size:
    :param C2_size:
    :param C3_size:
    :param D2_size:
    :param layer_name:
    :param is_training:
    :param pad:
    :return:
    """
    A1 = Conv2D(filters=A_size, kernel_size=1, strides=(1,1), padding=pad, activation="relu")(input)
    B1 = Conv2D(filters=B1_size, kernel_size=1, strides=(1,1), padding=pad, activation="relu")(input)
    B2 = Conv2D(filters=B2_size, kernel_size=3, strides=(1,1), padding=pad, activation="relu")(B1)
    C1 = Conv2D(filters=C1_size, kernel_size=1, strides=(1,1), padding=pad, activation="relu")(input)
    C2 = Conv2D(filters=C2_size, kernel_size=3, strides=(1,1), padding=pad, activation="relu")(C1)
    C3 = Conv2D(filters=C3_size, kernel_size=3, strides=(1,1), padding=pad, activation="relu")(C2)
    D1 = MaxPool2D(pool_size=3, strides=(1,1), padding=pad)(input)
    D2 = Conv2D(filters=D2_size, kernel_size=1, strides=(1,1), padding=pad, activation="relu")(D1)
    concat = Concatenate([A1, B2, C3, D2])

    return concat

def B_module(input, A_size, B1_size, B2_size, B3_size, C1_size, C2_size, C3_size, C4_size, C5_size, D2_size, pad="same"):
    """
    A module for Inception V2~3
    :param input:
    :param A_size:
    :param B1_size:
    :param B2_size:
    :param B3_size:
    :param C1_size:
    :param C2_size:
    :param C3_size:
    :param C4_size:
    :param C5_size:
    :param D2_size:
    :param layer_name:
    :param is_training:
    :param pad:
    :return:
    """
    A1 = Conv2D(filters=A_size, kernel_size=1, strides=(1,1), padding=pad, activation="relu")(input)
    B1 = Conv2D(filters=B1_size, kernel_size=1, strides=(1,1), padding=pad, activation="relu")(input)
    B2 = Conv2D(filters=B2_size, kernel_size=(1,3), strides=(1,1), padding=pad, activation="relu")(B1)
    B3 = Conv2D(filters=B3_size, kernel_size=(3,1), strides=(1,1), padding=pad, activation="relu")(B2)
    C1 = Conv2D(filters=C1_size, kernel_size=1, strides=(1,1), padding=pad, activation="relu")(input)
    C2 = Conv2D(filters=C2_size, kernel_size=(1,3), strides=(1,1), padding=pad, activation="relu")(C1)
    C3 = Conv2D(filters=C3_size, kernel_size=(3,1), strides=(1,1), padding=pad, activation="relu")(C2)
    C4 = Conv2D(filters=C4_size, kernel_size=(1,3), strides=(1,1), padding=pad, activation="relu")(C3)
    C5 = Conv2D(filters=C5_size, kernel_size=(3,1), strides=(1,1), padding=pad, activation="relu")(C4)
    D1 = MaxPool2D(pool_size=3, strides=(1,1), padding=pad)(input)
    D2 = Conv2D(filters=D2_size, kernel_size=1, strides=(1,1), padding=pad, activation="relu")(D1)
    concat = Concatenate([A1, B3, C5, D2])
    return concat

def C_module(input, A_size, B1_size, B2_size, B3_size, C1_size, C2_size, C3_size, C4_size, D2_size, pad="same"):
    """
    A module for Inception V2~3
    :param input:
    :param A_size:
    :param B1_size:
    :param B2_size:
    :param B3_size:
    :param C1_size:
    :param C2_size:
    :param C3_size:
    :param C4_size:
    :param D2_size:
    :param layer_name:
    :param is_training:
    :param pad:
    :return:
    """
    A1 = Conv2D(filters=A_size, kernel_size=1, strides=(1,1), padding=pad, activation="relu")(input)
    B1 = Conv2D(filters=B1_size, kernel_size=1, strides=(1,1), padding=pad, activation="relu")(input)
    B2 = Conv2D(filters=B2_size, kernel_size=(1,3), strides=(1,1), padding=pad, activation="relu")(B1)
    B3 = Conv2D(filters=B3_size, kernel_size=(3,1), strides=(1,1), padding=pad, activation="relu")(B2)
    C1 = Conv2D(filters=C1_size, kernel_size=1, strides=(1,1), padding=pad, activation="relu")(input)
    C2 = Conv2D(filters=C2_size, kernel_size=3, strides=(1,1), padding=pad, activation="relu")(C1)
    C3 = Conv2D(filters=C3_size, kernel_size=(1,3), strides=(1,1), padding=pad, activation="relu")(C2)
    C4 = Conv2D(filters=C4_size, kernel_size=(3,1), strides=(1,1), padding=pad, activation="relu")(C2)
    D1 = MaxPool2D(pool_size=3, strides=(1,1), padding=pad)(input)
    D2 = Conv2D(filters=D2_size, kernel_size=1, strides=(1,1), padding=pad, activation="relu")(D1)
    concat = Concatenate([A1, B3, C3, C4, D2])
    return concat

# XCEPTION
def X_module_entry(input, knum_out):
    """
    This module is for "Entry Flow" of Xception architecture.
    :param input : input tensor.
    :param knum_out : Output channels of separable convolution.
    :return : results
    """

    residual_connection = Conv2D(filters=knum_out, kernel_size=(1,1), strides=(2,2), padding="same")(input)
    residual_connection_B = BatchNormalization()(residual_connection)

    S1 = SeparableConv2D(filters=knum_out, kernel_size=(3,3), padding="same")(input)
    S_B1 = BatchNormalization()(S1)
    S_A1 = Activation(activation="relu")(S_B1)

    S2 = SeparableConv2D(filters=knum_out, kernel_size=(3,3), padding="same")(S_A1)
    S_B2 = BatchNormalization()(S2)
    S_A2 = Activation(actication="relu")(S_B2)

    Max = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")(S_A2)

    results = Add()([residual_connection_B, Max])

    return results

def X_module_middle(input, knum_out):
    """
    This module is for "Middle Flow" of Xception architecture.
    :param input : Input tensor.
    :param knum_out : Output channels of separable convolution.
    :return : results
    """

    residual_connection = Conv2D(filters=knum_out, kernel_size=(1,1), strides=(2,2), padding="same")(input)
    residual_connection_B = BatchNormalization()(residual_connection)

    S_A1 = Activation(activation="relu")(input)
    S1 = SeparableConv2D(filters=knum_out, kernel_size=(3,3), padding="same")(S_A1)
    S_B1 = BatchNormalization()(S1)

    S_A2 = Activation(activation="relu")(S_B1)
    S2 = SeparableConv2D(filters=knum_out, kernel_size=(3,3), padding="same")(S_A2)
    S_B2 = BatchNormalization()(S2)

    S_A3 = Activation(activation="relu")(S_B2)
    S3 = SeparableConv2D(filters=knum_out, kernel_size=(3,3), padding="same")(S_A3)
    S_B3 = BatchNormalization()(S3)

    results = Add()([residual_connection_B, S_B3])

    return results

def X_module_exit(input, knum_out1, knum_out2):
    """
    This module is for "Exit Flow" of Xception architecture.
    :param input: Input tensor.
    :param knum_out1: Output channels of 1st separable conv.
    :param knum_out2: Output channels of 2nd separable conv.
    :return: results.
    """

    residual_connection = Conv2D(filters=knum_out2, kernel_size=(1, 1), strides=(2, 2), padding="same")(input)
    residual_connection_B = BatchNormalization()(residual_connection)

    S_A1 = Activation(activation="relu")(input)
    S1 = SeparableConv2D(filters=knum_out1, kernel_size=(3,3), padding="same")(S_A1)
    S_B1 = BatchNormalization()(S1)

    S_A2 = Activation(activation="relu")(S_B1)
    S2 = SeparableConv2D(filters=knum_out2, kernel_size=(3,3), padding="same")(S_A2)
    S_B2 = BatchNormalization()(S2)

    Max = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same")(S_B2)

    results = Add()([residual_connection_B, Max])

    return results

def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs

""" deprecated version
def X_module_entry_DLV3P(input, knum, u_rate, c_rate, weight_decay, block_name):
    \"""
    This module is for Xception of DeepLabV3+.
    :param input: input tensor
    :param knum: number of kernels in this block
    :param u_rate: unit rate for three of atrous conv in this block. dtype = tuple. For example : u_rate = (1,2,4)
    :param c_rate: corresponding rate. dtype = scalar.
    :param block_name: block name.
    :return:
    \"""

    sep_conv1 = SeparableConv2D(filters=knum, kernel_size=(3,3), strides=(1,1), kernel_initializer="he_normal",
                                dilation_rate=u_rate[0]*c_rate, padding="same", name=block_name+"_sep_conv1", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(input)
    batch_conv1 = BatchNormalization()(sep_conv1)
    active_conv1 = Activation(activation="relu", name=block_name+"_act1")(batch_conv1)

    sep_conv2 = SeparableConv2D(filters=knum, kernel_size=(3,3), strides=(1,1), kernel_initializer="he_normal",
                                dilation_rate=u_rate[1]*c_rate, padding="same", name=block_name+"_sep_conv2", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(active_conv1)
    batch_conv2 = BatchNormalization()(sep_conv2)
    active_conv2 = Activation(activation="relu", name=block_name+"_act2")(batch_conv2)

    sep_conv3 = SeparableConv2D(filters=knum, kernel_size=(3,3), strides=(2,2), kernel_initializer="he_normal",
                                dilation_rate=u_rate[2]*c_rate, padding="same", name=block_name+"_sep_conv3", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(active_conv2)
    batch_conv3 = BatchNormalization()(sep_conv3)
    active_conv3 = Activation(activation="relu", name=block_name+"_act3")(batch_conv3)

    identity = Conv2D(filters=knum, kernel_size=(1,1), strides=(2,2), padding="same", name=block_name+"_identity", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(input)

    results = Add()([active_conv3, identity])

    return results

def X_module_middle_DLV3P(input, knum, u_rate, c_rate, weight_decay, block_name):
    \"""
    This module is for Xception of DeepLabV3+.
    :param input: input tensor
    :param knum: number of kernels in this block
    :param u_rate: unit rate for three of atrous conv in this block. dtype = tuple. For example : u_rate = (1,2,4)
    :param c_rate: corresponding rate. dtype = scalar.
    :param block_name: block name.
    :return:
    \"""

    sep_conv1 = SeparableConv2D(filters=knum, kernel_size=(3, 3), strides=(1, 1), kernel_initializer="he_normal",
                                dilation_rate=u_rate[0] * c_rate, padding="same", name=block_name + "_sep_conv1", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(input)
    batch_conv1 = BatchNormalization()(sep_conv1)
    active_conv1 = Activation(activation="relu")(batch_conv1)

    sep_conv2 = SeparableConv2D(filters=knum, kernel_size=(3, 3), strides=(1, 1), kernel_initializer="he_normal",
                                dilation_rate=u_rate[1] * c_rate, padding="same", name=block_name + "_sep_conv2", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(active_conv1)
    batch_conv2 = BatchNormalization()(sep_conv2)
    active_conv2 = Activation(activation="relu")(batch_conv2)

    sep_conv3 = SeparableConv2D(filters=knum, kernel_size=(3, 3), strides=(1, 1), kernel_initializer="he_normal",
                                dilation_rate=u_rate[2] * c_rate, padding="same", name=block_name + "_sep_conv3", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(active_conv2)
    batch_conv3 = BatchNormalization()(sep_conv3)
    active_conv3 = Activation(activation="relu")(batch_conv3)

    identity = input

    results = Add()([active_conv3, identity])

    return results

def X_module_exit_DLV3P(input, knum_in, knum_out, u_rate, c_rate, weight_decay, block_name):
    \"""

    :param input: This module is for Xception of DeepLabV3+.
    :param knum_in: number of kernels of first separable conv computation in this block
    :param knum_out: number of kernels of second / third separable conv computation in this block
    :param u_rate: unit rate for three of atrous conv in this block. dtype = tuple. For example : u_rate = (1,2,4)
    :param c_rate: corresponding rate. dtype = scalar.
    :param block_name:block name.
    :return:
    \"""

    sep_conv1 = SeparableConv2D(filters=knum_in, kernel_size=(3,3), kernel_initializer="he_normal",
                                strides=(1,1), dilation_rate=u_rate[0]*c_rate, padding="same", name=block_name+"_sep_conv1", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(input)
    batch_conv1 = BatchNormalization()(sep_conv1)
    active_conv1 = Activation(activation="relu")(batch_conv1)

    sep_conv2 = SeparableConv2D(filters=knum_out, kernel_size=(3,3), kernel_initializer="he_normal",
                                strides=(1,1), dilation_rate=u_rate[1]*c_rate, padding="same", name=block_name+"_sep_conv2", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(active_conv1)
    batch_conv2 = BatchNormalization()(sep_conv2)
    active_conv2 = Activation(activation="relu")(batch_conv2)

    sep_conv3 = SeparableConv2D(filters=knum_out, kernel_size=(3,3), kernel_initializer="he_normal",
                                strides=(2,2), dilation_rate=u_rate[2]*c_rate, padding="same", name=block_name+"_sep_conv3", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(active_conv2)
    batch_conv3 = BatchNormalization()(sep_conv3)
    active_conv3 = Activation(activation="relu")(batch_conv3)

    identity = Conv2D(filters=knum_out, kernel_size=(1,1), strides=(2,2), padding="same", name=block_name+"_identity", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(input)

    results = Add()([active_conv3, identity])

    return results
"""
#RESNET
def Resblock(input, knum, layer_name, pad="same", verbose=False):

    #identity mapping
    identity = input
    if verbose :
        identity = MaxPool2D(pool_size=1, strides=2)(identity)
        zero_pad = K.zeros_like(identity)
        identity = Concatenate()([identity, zero_pad])

    if not verbose :
        Conv_L1 = Conv2D(filters=knum, kernel_size=3, kernel_initializer="he_normal",
                         strides=1, padding=pad, name=layer_name+"_C_L1")(input)
    else :
        Conv_L1 = Conv2D(filters=knum, kernel_size=3, kernel_initializer="he_normal",
                         strides=2, padding=pad, name=layer_name+"_C_L1")(input)
    BN_L1 = BatchNormalization()(Conv_L1)
    AC_L1 = Activation(activation="relu")(BN_L1)

    Conv_L2 = Conv2D(filters=knum, kernel_size=3, kernel_initializer="he_normal",
                     strides=1, padding=pad, name=layer_name+"_C_L2")(AC_L1)
    BN_L2 = BatchNormalization()(Conv_L2)

    #shortcut
    shortcut = Add()([BN_L2, identity])
    shortcut = Activation(activation="relu")(shortcut)

    return shortcut

def Resblock_bn(input, knum_in, knum_out, layer_name, dilation=(1,1), pad="same", verbose=False):
    """
    Residual block bottle neck version - module for over ResNet50
    :param input: input feature
    :param knum_in: the number of filters(kernels, or size of output feature) of Conv_L1 and Conv_L2
    :param knum_out: the number of filters(kernels, or size of output feature) of Conv_L3
    :param layer_name: Module name
    :param pad: "same" or "valid" ( identical with Tensorflow old version(1.x) )
    :param verbose: reduce heightxwidth(verbose=True) or not(verbose=False)
    """
    # identity mapping
    identity = input

    if not verbose : Conv_L1 = Conv2D(filters=knum_in,
                                      kernel_size=(1,1),
                                      kernel_initializer="he_normal",
                                      strides=(1,1),
                                      padding=pad,
                                      dilation_rate=dilation,
                                      name=layer_name+"_Conv_L1")(input)
    else : Conv_L1 = Conv2D(filters=knum_in,
                            kernel_size=(1,1),
                            strides=(2,2),
                            kernel_initializer="he_normal",
                            padding=pad,
                            dilation_rate=dilation,
                            name=layer_name+"_Conv_L1")(input)
    BN1 = BatchNormalization()(Conv_L1)
    AC1 = Activation(activation="relu")(BN1)

    Conv_L2 = Conv2D(filters=knum_in,
                     kernel_size=(3,3),
                     kernel_initializer="he_normal",
                     strides=(1,1),
                     padding=pad,
                     dilation_rate=dilation,
                     name=layer_name+"_Conv_L2")(AC1)
    BN2 = BatchNormalization()(Conv_L2)
    AC2 = Activation(activation="relu")(BN2)

    Conv_L3 = Conv2D(filters=knum_out,
                     kernel_size=(1,1),
                     kernel_initializer="he_normal",
                     strides=(1,1),
                     padding=pad,
                     dilation_rate=dilation,
                     name=layer_name+"_Conv_L3")(AC2)
    BN3 = BatchNormalization()(Conv_L3)

    if not verbose : identity = Conv2D(filters=knum_out,
                                       kernel_size=(1,1),
                                       kernel_initializer="he_normal",
                                       strides=(1,1),
                                       dilation_rate=dilation,
                                       padding=pad)(identity)
    else : identity = Conv2D(filters=knum_out,
                             kernel_size=(1,1),
                             kernel_initializer="he_normal",
                             strides=(2,2),
                             dilation_rate=dilation,
                             padding=pad)(identity)

    # shortcuts
    Shortcut = Add()([BN3, identity])
    Shortcut = Activation(activation="relu")(Shortcut)

    return Shortcut

# D-LINKNET
def Dilated_layers(input, knum, ksize, layer_name, pad="same"):
    """
    Dilated layers for DlinkNet
    :param input:
    :param knum:
    :param ksize:
    :param layer_name:
    :param pad:
    :return:
    """

    input_identity = input

    # layer 1
    D1_1 = Conv2D(filters=knum, kernel_size=ksize, kernel_initializer="he_normal",
                  activation="relu", strides=1, padding=pad, dilation_rate=1, name=layer_name+"_D1_1")(input)
    D1_2 = Conv2D(filters=knum, kernel_size=ksize, kernel_initializer="he_normal",
                  activation="relu", strides=1, padding=pad, dilation_rate=2, name=layer_name+"_D1_2")(D1_1)
    D1_3 = Conv2D(filters=knum, kernel_size=ksize, kernel_initializer="he_normal",
                  activation="relu", strides=1, padding=pad, dilation_rate=4, name=layer_name+"_D1_3")(D1_2)
    D1_4 = Conv2D(filters=knum, kernel_size=ksize, kernel_initializer="he_normal",
                  activation="relu", strides=1, padding=pad, dilation_rate=8, name=layer_name+"_D1_4")(D1_3)

    # layer 2
    D2_1 = Conv2D(filters=knum, kernel_size=ksize, kernel_initializer="he_normal",
                  activation="relu", strides=1, padding=pad, dilation_rate=1, name=layer_name+"_D2_1")(input)
    D2_2 = Conv2D(filters=knum, kernel_size=ksize, kernel_initializer="he_normal",
                  activation="relu", strides=1, padding=pad, dilation_rate=2, name=layer_name+"_D2_2")(D2_1)
    D2_3 = Conv2D(filters=knum, kernel_size=ksize, kernel_initializer="he_normal",
                  activation="relu", strides=1, padding=pad, dilation_rate=4, name=layer_name+"_D2_3")(D2_2)

    # layer 3
    D3_1 = Conv2D(filters=knum, kernel_size=ksize, kernel_initializer="he_normal",
                  activation="relu", strides=1, padding=pad, dilation_rate=1, name=layer_name + "_D3_1")(input)
    D3_2 = Conv2D(filters=knum, kernel_size=ksize, kernel_initializer="he_normal",
                  activation="relu", strides=1, padding=pad, dilation_rate=2, name=layer_name + "_D3_2")(D3_1)

    # layer 4
    D4_1 = Conv2D(filters=knum, kernel_size=ksize, kernel_initializer="he_normal",
                  activation="relu", strides=1, padding=pad, dilation_rate=1, name=layer_name + "_D4_1")(input)

    # Returns
    out_tail = Add()([input_identity, D1_4, D2_3, D3_2, D4_1])

    return out_tail

def Transblock(input, knum_out, layer_name, pad="same"):
    """
    DlinkNet Transposed Conv layers.
    :param input:
    :param knum_out:
    :param layer_name:
    :param pad:
    :return:
    """

    tensor_shape = K.int_shape(input)
    Dim_reduce = Conv2D(filters=int(tensor_shape[-1]/4), kernel_size=1, kernel_initializer="he_normal",
                        activation="relu", strides=1, padding=pad, name=layer_name+"dim_reduce")(input)
    Trans_conv = Conv2DTranspose(filters=int(tensor_shape[-1]/4), kernel_size=3, kernel_initializer="he_normal",
                                 activation="relu", strides=2, padding=pad, name=layer_name+"_transpose")(Dim_reduce)
    Dim_expand = Conv2D(filters=knum_out, kernel_size=1, kernel_initializer="he_normal",
                        activation="relu", strides=1, padding=pad, name=layer_name+"dim_expand")(Trans_conv)

    return Dim_expand

# DENSENET
def Denseblock(input, L, k, block_name, pad="same") :
    """

    :param input:
    :param L: the number of iteration (how many blocks are needed?)
    :param k: k is "growth rate" (filter num)
    :return:
    """
    for i in range(L) :
        bn1x1 = BatchNormalization()(input)
        relu1x1 = Activation(activation="relu")(bn1x1)
        conv1x1 = Conv2D(filters=4*k, kernel_size=(1,1), kernel_initializer="he_normal",
                         strides=(1,1), padding=pad, name=block_name+"_conv1x1_"+str(i+1))(relu1x1)
        bn3x3 = BatchNormalization()(conv1x1)
        relu3x3 = Activation(activation="relu")(bn3x3)
        conv3x3 = Conv2D(filters=k, kernel_size=(3,3), kernel_initializer="he_normal",
                         strides=(1,1), padding=pad, name=block_name+"_conv3x3_"+str(i+1))(relu3x3)

        concat = Concatenate(axis=-1)([input, conv3x3])
        input = concat

    return input

def transition_layer(input, layer_name, theta=0.5, pad="same") :
    """

    :param input: input feature.
    :param theta: Compression rate of input channel depth.
    :return:
    """
    input_channel = input.shape[-1]
    bn1x1 = BatchNormalization()(input)
    relu1x1 = Activation(activation="relu")(bn1x1)
    conv1x1 = Conv2D(filters=tf.math.floor(input_channel*theta), kernel_size=(1,1), kernel_initializer="he_normal",
                     strides=(1,1), padding=pad, name=layer_name+"conv1x1")(relu1x1)
    avg_pool2x2 = AvgPool2D(pool_size=(2,2), strides=(2,2), padding=pad)(conv1x1)
    return avg_pool2x2

# Hiearachical CONCEPT
def hierachical_weights(tensor, category, ratio=0.5):
    """
    :param tensor:
    :param category: 2 Options, "main2mid", "mid2det".
                     "main2mid" : repeat list - [6, 5, 3, 2, 2, 2, 2, 1]
                     "mid2det" : repeat list - [2, 1, 2, 1, 5, 3, 2, 2, 1, 1, 2, 1, 1, 1, 1, 3, 1, 2, 3, 3, 2, 1, 1]
    :return:
    """
    if category == "main2mid" : repeat_list = [6, 5, 3, 2, 2, 2, 2, 1]
    elif category == "mid2det" : repeat_list = [2, 1, 2, 1, 5, 3, 2, 2, 1, 1, 2, 1, 1, 1, 1, 3, 1, 2, 3, 3, 2, 1, 1]
    else : raise SyntaxError("category option should be one of 'main2mid' or 'mid2det'. Please check hierachical weights def input.")

    duplicate = tf.repeat(tensor, repeats=repeat_list, axis=-1)
    return duplicate*ratio

def hierachical_weight_table(num_classes, weight_a, weight_b, weight_c, weight_d) :
    """

    :param num_classes: the number of class, which is index of column and row
    :param weight_a: a weight value when matched.
    :param weight_b: a weight value when not matched, but middle category is correct.
    :param weight_c: a weight value when not matched, but main category is correct.
    :param weight_d: when totally not matched.
    :return: hierachical wieght table
    """
    hierachical_weight_table = np.zeros([num_classes, num_classes])
    main_category_relation = {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0,
                              5 : 0, 6 : 0, 7 : 0, 8 : 0, 9 : 0,
                              10 : 0, 11 : 0, 12 : 0, 13 : 0, 14 : 1,
                              15 : 1, 16 : 1, 17 : 1, 18 : 1, 19 : 1,
                              20 : 1, 21 : 1, 22 : 2, 23 : 2, 24 : 2,
                              25 : 3, 26 : 3, 27 : 3, 28 : 3, 29 : 4,
                              30 : 4, 31 : 4, 32 : 5, 33 : 5, 34 : 5,
                              35 : 5, 36 : 5, 37 : 5, 38 : 6, 39 : 6,
                              40 : 6, 41 : 7}
    middle_category_relation = {0 : 0, 1 : 0, 2 : 1, 3 : 2, 4 : 2,
                                5 : 3, 6 : 4, 7 : 4, 8 : 4, 9 : 4,
                                10 : 4, 11 : 5, 12 : 5, 13 : 5, 14 : 6,
                                15 : 6, 16 : 7, 17 : 7, 18 : 8, 19 : 9,
                                20 : 10, 21 : 10, 22 : 11, 23 : 12, 24 : 13,
                                25 : 14, 26 : 15, 27 : 15, 28 : 15, 29 : 16,
                                30 : 17, 31 : 17, 32 : 18, 33 : 18, 34 : 18,
                                35 : 19, 36 : 19, 37 : 19, 38 : 20, 39 : 20,
                                40 : 21, 41 : 22}

    for i in range(num_classes) :
        for j in range(num_classes) :
            if i==j : hierachical_weight_table[i,j] = weight_a
            elif i!=j :
                if middle_category_relation[i] == middle_category_relation[j] : hierachical_weight_table[i,j] = weight_b
                elif main_category_relation[i] == main_category_relation[j] : hierachical_weight_table[i,j] = weight_c
                else : hierachical_weight_table[i,j] = weight_d

    return hierachical_weight_table

def to_main_category(tensor, to_category):
    """
    This code is for testing middle, detail category of landcover at identical level of Main category.
    :param tensor: softmax result of middle, detail category
    :param to_category: 2 options - "mid2main", "det2main"
    :return:
    """

    if to_category == "mid2main":
        c1 = K.expand_dims(K.max(tensor[:, :, :, 0:6], axis=-1))
        c2 = K.expand_dims(K.max(tensor[:, :, :, 6:11], axis=-1))
        c3 = K.expand_dims(K.max(tensor[:, :, :, 11:14], axis=-1))
        c4 = K.expand_dims(K.max(tensor[:, :, :, 14:16], axis=-1))
        c5 = K.expand_dims(K.max(tensor[:, :, :, 16:18], axis=-1))
        c6 = K.expand_dims(K.max(tensor[:, :, :, 18:20], axis=-1))
        c7 = K.expand_dims(K.max(tensor[:, :, :, 20:22], axis=-1))
        c8 = K.expand_dims(tensor[:, :, :, 22])

    elif to_category == "det2main" :
        c1 = K.expand_dims(K.max(tensor[:, :, :, 0:14], axis=-1))
        c2 = K.expand_dims(K.max(tensor[:, :, :, 14:22], axis=-1))
        c3 = K.expand_dims(K.max(tensor[:, :, :, 22:25], axis=-1))
        c4 = K.expand_dims(K.max(tensor[:, :, :, 25:29], axis=-1))
        c5 = K.expand_dims(K.max(tensor[:, :, :, 29:32], axis=-1))
        c6 = K.expand_dims(K.max(tensor[:, :, :, 32:38], axis=-1))
        c7 = K.expand_dims(K.max(tensor[:, :, :, 38:41], axis=-1))
        c8 = K.expand_dims(tensor[:, :, :, 41])
    else :
        raise SyntaxError("'to_category' values have two options : 'mid2main' or 'det2main'. please check 'to_category' value.")

    return Concatenate(axis=-1)([c1, c2, c3, c4, c5, c6, c7, c8])

# DeepLabV3
def Resblock_DLV3(input, f1, f2, f3, verbose, block_name, pad="same"):
    """
    This block is for front parts, which don't contain atrous convolution.
    :param input: input tensor
    :param f1: number of filters of conv1 computation
    :param f2: number of filters of conv2 computation
    :param f3: number of filters of conv3 computation
    :param verbose: whether this block is the last one(True) or not(False)
    :param block_name: block name
    :param pad: padding, default = "same"
    :return:
    """
    conv1 = Conv2D(filters=f1, kernel_size=(3,3), strides=(1,1),
                   kernel_initializer="he_normal", padding=pad, name=block_name+"_conv1")(input)
    conv1_batch = BatchNormalization()(conv1)
    conv1_active = Activation(activation="relu")(conv1_batch)

    conv2 = Conv2D(filters=f2, kernel_size=(3,3), strides=(1,1),
                   kernel_initializer="he_normal", padding=pad, name=block_name+"_conv2")(conv1_active)
    conv2_batch = BatchNormalization()(conv2)
    conv2_active = Activation(activation="relu")(conv2_batch)

    if verbose == "True" :
        conv3 = Conv2D(filters=f3, kernel_size=(3,3), strides=(1,1),
                       kernel_initializer="he_normal", padding=pad, name=block_name+"_conv3")(conv2_active)
        identity = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1,1),
                          kernel_initializer="he_normal", padding="valid", name=block_name + "_identity")(input)
    else :
        conv3 = Conv2D(filters=f3, kernel_size=(3,3), strides=(2,2),
                       kernel_initializer="he_normal", padding=pad, name=block_name+"_conv3")(conv2_active)
        identity = Conv2D(filters=f3, kernel_size=(1, 1), strides=(2,2),
                          kernel_initializer="he_normal", padding="valid", name=block_name + "_identity")(input)

    conv3_batch = BatchNormalization()(conv3)
    conv3_active = Activation(activation="relu")(conv3_batch)

    results = Add()([conv3_active, identity])
    return results

def Resblock_DLV3_atrous(input, f1, f2, f3, verbose, u_rate, c_rate, block_name, pad="same"):
    """
    This block is for front parts, which contain atrous convolution.
    :param input: input tensor
    :param f1: number of filters of conv1 computation
    :param f2: number of filters of conv2 computation
    :param f3: number of filters of conv3 computation
    :param verbose: whether this block is the last one(True) or not(False)
    :param block_name: block name
    :param u_rate : the unit rate of multi grid, dtype = tuple. Example : (1,2,4)
    :param c_rate : the corresponding rate of block, dtype = scalar. Example : 2
    :param pad: padding, default = "same"
    :return:
    """
    conv1 = Conv2D(filters=f1, kernel_size=(3,3), strides=(1,1), padding=pad,
                   dilation_rate=u_rate[0]*c_rate, kernel_initializer="he_normal", name=block_name+"_conv1")(input)
    conv1_batch = BatchNormalization()(conv1)
    conv1_active = Activation(activation="relu")(conv1_batch)

    conv2 = Conv2D(filters=f2, kernel_size=(3,3), strides=(1,1), padding=pad,
                   dilation_rate=u_rate[1]*c_rate, kernel_initializer="he_normal", name=block_name+"_conv2")(conv1_active)
    conv2_batch = BatchNormalization()(conv2)
    conv2_active = Activation(activation="relu")(conv2_batch)

    conv3 = Conv2D(filters=f3, kernel_size=(3,3), strides=(1,1), padding=pad,
                   dilation_rate=u_rate[2]*c_rate, kernel_initializer="he_normal", name=block_name+"_conv3")(conv2_active)
    conv3_batch = BatchNormalization()(conv3)
    conv3_active = Activation(activation="relu")(conv3_batch)

    identity = Conv2D(filters=f3, kernel_size=(1,1), strides=(1,1), padding=pad)(input)

    results = Add()([conv3_active, identity])
    return results

# PSPNet
def Pyramid_module(input) :
    """
    REF : Hengshuang Zhao etal, <Pyramid Scene Parsing Network>, 2017.
    This module is "Pyramid scene parsing module" which is mentioned on the paper.
    shape of input tensor is [batch_size, 512, 512, channel]
    :return: 
    """
    input_shape = input.shape[1:] # [batch_size, 16, 16, 2048]
    pyramid_level = 4 # outputsize : 1, 2, 4, 8

    pool_feature_1 = GlobalAvgPool2D()(input)
    pool_feature_2 = AvgPool2D(pool_size=10, strides=5, padding="valid")(input)
    pool_feature_4 = AvgPool2D(pool_size=6, strides=3, padding="valid")(input)
    pool_feature_8 = AvgPool2D(pool_size=2, strides=2, padding="valid")(input)

    pool_feature_1 = K.expand_dims(pool_feature_1)
    pool_feature_1 = K.expand_dims(pool_feature_1)
    pool_feature_1 = tf.transpose(pool_feature_1, [0,3,2,1])\

    f1_conv1x1 = Conv2D(filters=int(input_shape[-1]/pyramid_level), kernel_size=(1,1))(pool_feature_1)
    pool_feature_1 = UpSampling2D(size=(16,16), interpolation="bilinear")(f1_conv1x1)
    f2_conv1x1 = Conv2D(filters=int(input_shape[-1] / pyramid_level), kernel_size=(1,1))(pool_feature_2)
    pool_feature_2 = UpSampling2D(size=(8, 8), interpolation="bilinear")(f2_conv1x1)
    f3_conv1x1 = Conv2D(filters=int(input_shape[-1] / pyramid_level), kernel_size=(1,1))(pool_feature_4)
    pool_feature_4 = UpSampling2D(size=(4, 4), interpolation="bilinear")(f3_conv1x1)
    f4_conv1x1 = Conv2D(filters=int(input_shape[-1] / pyramid_level), kernel_size=(1,1))(pool_feature_8)
    pool_feature_8 = UpSampling2D(size=(2, 2), interpolation="bilinear")(f4_conv1x1)

    output = Concatenate()([input, pool_feature_1, pool_feature_2, pool_feature_4, pool_feature_8])
    return output

# SEMANTIC SEGMENTATION PREDICTION - 2020 토지피복도용 COLOR TABLE
def class2color(mask, name):
    # mask = mask.reshape(1024, 1024, 28)
    # mask = np.argmax(mask, axis=-1)
    mask_zero = np.zeros([1024, 1024, 3])

    for class_num in range(28):
        mask_label = mask == class_num

        if class_num == 0:
            mask_zero[mask_label] = [194, 230, 254]
        elif class_num == 1:
            mask_zero[mask_label] = [11, 193, 223]
        elif class_num == 2:
            mask_zero[mask_label] = [132, 132, 192]
        elif class_num == 3:
            mask_zero[mask_label] = [80, 100, 180]
        elif class_num == 4:
            mask_zero[mask_label] = [138, 113, 246]

        elif class_num == 5:
            mask_zero[mask_label] = [50, 140, 240]
        elif class_num == 6:
            mask_zero[mask_label] = [110, 130, 250]
        elif class_num == 7:
            mask_zero[mask_label] = [91, 255, 255]
        elif class_num == 8:
            mask_zero[mask_label] = [168, 230, 244]
        elif class_num == 9:
            mask_zero[mask_label] = [102, 249, 247]

        elif class_num == 10:
            mask_zero[mask_label] = [10, 228, 245]
        elif class_num == 11:
            mask_zero[mask_label] = [115, 220, 223]
        elif class_num == 12:
            mask_zero[mask_label] = [44, 177, 184]
        elif class_num == 13:
            mask_zero[mask_label] = [18, 145, 184]
        elif class_num == 14:
            mask_zero[mask_label] = [0, 100, 170]

        elif class_num == 15:
            mask_zero[mask_label] = [100, 255, 0]
        elif class_num == 16:
            mask_zero[mask_label] = [148, 213, 161]
        elif class_num == 17:
            mask_zero[mask_label] = [50, 150, 100]
        elif class_num == 18:
            mask_zero[mask_label] = [208, 167, 180]
        elif class_num == 19:
            mask_zero[mask_label] = [153, 116, 153]

        elif class_num == 20:
            mask_zero[mask_label] = [162, 30, 124]
        elif class_num == 21:
            mask_zero[mask_label] = [160, 150, 130]
        elif class_num == 22:
            mask_zero[mask_label] = [138, 90, 88]
        elif class_num == 23:
            mask_zero[mask_label] = [172, 181, 123]
        elif class_num == 24:
            mask_zero[mask_label] = [255, 242, 159]

        elif class_num == 25:
            mask_zero[mask_label] = [255, 167, 62]
        elif class_num == 26:
            mask_zero[mask_label] = [255, 109, 93]
        elif class_num == 27:
            mask_zero[mask_label] = [255, 57, 23]
        elif class_num == 28:
            mask_zero[mask_label] = [0, 0, 0]

        # if class_num == 0:
        #     mask_zero[mask_label] = [194, 230, 254]
        # elif class_num == 1:
        #     mask_zero[mask_label] = [111, 193, 223]
        # elif class_num == 2:
        #     mask_zero[mask_label] = [132, 132, 192]
        # elif class_num == 3:
        #     mask_zero[mask_label] = [184, 131, 237]
        # elif class_num == 4:
        #     mask_zero[mask_label] = [164, 176, 223]
        # elif class_num == 5:
        #     mask_zero[mask_label] = [138, 113, 246]
        # elif class_num == 6:
        #     mask_zero[mask_label] = [254, 38, 229]
        # elif class_num == 7:
        #     mask_zero[mask_label] = [81, 50, 197]
        # elif class_num == 8:
        #     mask_zero[mask_label] = [78, 4, 252]
        # elif class_num == 9:
        #     mask_zero[mask_label] = [42, 65, 247]
        # elif class_num == 10:
        #     mask_zero[mask_label] = [0, 0, 115]
        # elif class_num == 11:
        #     mask_zero[mask_label] = [18, 177, 246]
        # elif class_num == 12:
        #     mask_zero[mask_label] = [0, 122, 255]
        # elif class_num == 13:
        #     mask_zero[mask_label] = [27, 88, 199]
        # elif class_num == 14:
        #     mask_zero[mask_label] = [191, 255, 255]
        # elif class_num == 15:
        #     mask_zero[mask_label] = [168, 230, 244]
        # elif class_num == 16:
        #     mask_zero[mask_label] = [102, 249, 247]
        # elif class_num == 17:
        #     mask_zero[mask_label] = [10, 228, 245]
        # elif class_num == 18:
        #     mask_zero[mask_label] = [115, 220, 223]
        # elif class_num == 19:
        #     mask_zero[mask_label] = [44, 177, 184]
        # elif class_num == 20:
        #     mask_zero[mask_label] = [18, 145, 184]
        # elif class_num == 21:
        #     mask_zero[mask_label] = [0, 100, 170]
        # elif class_num == 22:
        #     mask_zero[mask_label] = [44, 160, 51]
        # elif class_num == 23:
        #     mask_zero[mask_label] = [64, 79, 10]
        # elif class_num == 24:
        #     mask_zero[mask_label] = [51, 102, 51]
        # elif class_num == 25:
        #     mask_zero[mask_label] = [148, 213, 161]
        # elif class_num == 26:
        #     mask_zero[mask_label] = [90, 228, 128]
        # elif class_num == 27:
        #     mask_zero[mask_label] = [90, 176, 113]
        # elif class_num == 28:
        #     mask_zero[mask_label] = [51, 126, 96]
        # elif class_num == 29:
        #     mask_zero[mask_label] = [208, 167, 180]
        # elif class_num == 30:
        #     mask_zero[mask_label] = [153, 116, 153]
        # elif class_num == 31:
        #     mask_zero[mask_label] = [162, 30, 124]
        # elif class_num == 32:
        #     mask_zero[mask_label] = [236, 219, 193]
        # elif class_num == 33:
        #     mask_zero[mask_label] = [202, 197, 171]
        # elif class_num == 34:
        #     mask_zero[mask_label] = [165, 182, 171]
        # elif class_num == 35:
        #     mask_zero[mask_label] = [138, 90, 88]
        # elif class_num == 36:
        #     mask_zero[mask_label] = [172, 181, 123]
        # elif class_num == 37:
        #     mask_zero[mask_label] = [255, 242, 159]
        # elif class_num == 38:
        #     mask_zero[mask_label] = [255, 167, 62]
        # elif class_num == 39:
        #     mask_zero[mask_label] = [255, 109, 93]
        # elif class_num == 40:
        #     mask_zero[mask_label] = [255, 57, 23]
        # elif class_num == 41:
        #     mask_zero[mask_label] = [0, 0, 0]
        # else:
        #     print("Error")
    print(name)
    cv2.imwrite("E:\\Data_list\\2020_Landcover\\Data\\mask_mod_img\\" + name, mask_zero.astype(np.uint8))

""" ViT : 2021.06.03. / SETR : 2021.06.03. ~ 2021.06.17. """

def sequentializer_conv(input_tensor,
                        patch_size,
                        embedding_dim):
    seq = Conv2D(filters=embedding_dim,
                 kernel_size=patch_size,
                 strides=patch_size,
                 padding="valid")(input_tensor)
    seq = tf.reshape(seq,
                     shape=[-1, seq.shape[1]*seq.shape[2], seq.shape[3]])
    return seq

def sequentializer_patch(input_tensor,
                         patch_size,
                         embedding_dim):
    seq = tf.image.extract_patches(images=input,
                                   size=[1, patch_size, patch_size, 1],
                                   strides=[1, patch_size, patch_size, 1],
                                   rates=[1, 1, 1, 1],
                                   padding="valid")(input_tensor)
    seq = tf.reshape(seq,
                     shape=[-1, seq.shape[1]*seq.shape[2], seq.shape[3]])
    seq = Dense(units=embedding_dim)(seq)
    return seq

def MSA(input_tensor,
        hidden_size,
        num_of_head):

    projection_dim = hidden_size//num_of_head

    # query
    query = Dense(units=hidden_size)(input_tensor)
    multi_head_query = tf.reshape(query,
                                  shape=[-1, query.shape[1], num_of_head, projection_dim])
    multi_head_query = tf.transpose(multi_head_query,
                                    perm=[0, 2, 1, 3]) # [batch, num_of_head, seq_length, project_dim]

    # key
    key = Dense(units=hidden_size)(input_tensor)
    multi_head_key = tf.reshape(key,
                                shape=[-1, key.shape[1], num_of_head, projection_dim])
    multi_head_key = tf.transpose(multi_head_key,
                                  perm=[0, 2, 1, 3]) # [batch, num_of_head, seq_length, project_dim]

    # value
    value = Dense(units=hidden_size)(input_tensor)
    multi_head_value = tf.reshape(value,
                                  shape=[-1, value.shape[1], num_of_head, projection_dim])
    multi_head_value = tf.transpose(multi_head_value,
                                    perm=[0, 2, 1, 3]) # [batch, num_of_head, seq_length, project_dim]
    score = tf.matmul(multi_head_query, multi_head_key, transpose_b=True) # [batch, num_of_head, seq_length, seq_length]
    scale = tf.cast(tf.shape(key)[-1], score.dtype)
    scaled_score = score / tf.math.sqrt(scale)
    attention = tf.keras.activations.softmax(scaled_score, axis=-1) # [batch, num_of_head, seq_length, seq_length]
    output = tf.matmul(attention, multi_head_value) # [batch, num_of_head, seq_length, project_dim]
    output = tf.transpose(output,
                          perm=[0, 2, 1, 3])

    # output = tf.transpose(output, [0, 2, 1, 3])
    concat_output = tf.reshape(output,
                               shape=[-1, output.shape[1], hidden_size])
    # concat dense
    output = Dense(units=hidden_size)(concat_output)

    return output

def MLP(input_tensor,
        hidden_size):

    dn1 = Dense(units=hidden_size)(input_tensor)
    dp = Dropout(rate=0.2)(dn1)
    gl = Activation(activation="gelu")(dp)
    dn2 = Dense(units=hidden_size)(gl)

    return dn2

def Transformer(input_tensor,
                num_heads):

    hidden_size = input_tensor.shape[-1]
    # Key, Query, Value
    ly1 = LayerNormalization()(input_tensor)
    msa = MSA(input_tensor=ly1,
              num_of_head=num_heads,
              hidden_size=hidden_size)
    rb1 = ly1 + msa

    ly2 = LayerNormalization()(rb1)
    mlp = MLP(input_tensor=ly2,
              hidden_size=hidden_size)
    rb2 = mlp + rb1

    return rb2

def positionembds(input_shape):
    position_embd = tf.Variable(
        initial_value=tf.random_normal_initializer(stddev=0.06)(
            shape=(1, input_shape[1], input_shape[2])
            ),
            dtype="float32",
            trainable=True
        )
    return position_embd

def classtoken(input_tensor):
    hidden_size = input_tensor.shape[-1]
    cls_init = tf.zeros_initializer()
    cls = tf.Variable(
            initial_value=cls_init(shape=[1, 1, hidden_size], dtype="float32"),
            trainable=True
        )
    # cls_broadcasted = tf.cast(
    #         tf.broadcast_to(cls, [1, 1, hidden_size]),
    #         dtype=cls.dtype
    #     )
    return cls

# definition reference : github.co
# m/faustomorales/vit-keras/blob/master/vit_keras/layers.py
class PositionEmbds(Layer):
    def build(self, input_shape):
        self.position_en=tf.Variable(
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, input_shape[1], input_shape[2])
            ),
            dtype="float32",
            trainable=True
        )
    def call(self, inputs):
        return inputs + tf.cast(self.pe, dtype=inputs.dtype)

class ClassToken(Layer):
    def build(self, input_shape):
        self.hidden_size = input_shape[-1] # the num of channel
        self.cls = tf.Variable(
            initial_value=tf.zeros_initializer(input_shape=(1, 1, self.hidden_size), # 3dim (batch, height, width)
                                               dtype="float32"),
            trainable=True
        )
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype
        )
        return tf.concat([cls_broadcasted, inputs], axis=1) # top of height

"""STANet module : 2021.06.17."""

def bam(input_tensor,
        name="bam_") :
    """
    두 시점의 피쳐가 동시에 conv 연산이 돼서는 안된다.
    :param input_tensor: input_tensor : [batch, height, width*2, channel]
    :param name:
    :return:
    """

    # K, Q, V
    x = input_tensor
    h, w, c = x.shape[1], x.shape[2], x.shape[3] # h=h, w=2w, c=c

    Q = Conv2D(filters=c//8,
               kernel_size=1,
               strides=1,
               padding="valid")(x)
    Q = Activation(activation="gelu")(Q)
    Q = BatchNormalization()(Q)
    Q = tf.reshape(Q,
                   shape=[-1, h*w, c//8])

    K = Conv2D(filters=c//8,
               kernel_size=1,
               strides=1,
               padding="valid")(x)
    K = Activation(activation="gelu")(K)
    K = BatchNormalization()(K)
    K = tf.reshape(K,
                   shape=[-1, h*w, c//8])

    V = Conv2D(filters=c,
               kernel_size=1,
               strides=1,
               padding="valid")(x)
    V = Activation(activation="gelu")(V)
    V = BatchNormalization()(V)
    V = tf.reshape(V,
                   shape=[-1, h*w, c])

    attention = tf.matmul(Q, tf.transpose(K, perm=[0, 2, 1]))/tf.math.sqrt(tf.cast(c//8, tf.float32))
    attention = tf.keras.activations.softmax(attention, axis=-1)

    attention_score = tf.matmul(attention, V)
    attention_score = tf.reshape(attention_score,
                                 shape=[-1, h, w, c])
    bam = attention_score + x

    return bam

def pam(input_tensor,
        k_channels,
        v_channels,
        scales = (1,2,4,8),
        d_sample = 1):
    """

    :param input_tensor: input tensor shape = [Batch, H, 2W, C]
                         Concatenated 2 temporal tensor along W axis.
    :param k_channels: Key channel(=Query channel)
    :param v_channels: Value channel
    :param scales: default = (1, 2, 4, 8)
    :param d_sample: default = 1
    :return:
    """

    context_list = []
    for scale in scales :

        # input shape = [b, h, w*2, c]
        if d_sample != 1 : input_tensor = AvgPool2D(pool_size=(d_sample, d_sample))(input_tensor)
        b, h, w, c = input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2] // 2, input_tensor.shape[3]

        local_y = []
        local_x = []
        step_h, step_w = h // scale, w // scale
        for i in range(0, scale):
            for j in range(0, scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, h), min(start_y + step_w, w)
                if i == (scale - 1):
                    end_x = h
                if j == (scale - 1):
                    end_y = w
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]

        # Key, Query, Value
        K = Conv2D(filters=k_channels,
                   kernel_size=1,
                   strides=1,
                   padding="valid")(input_tensor)
        K = Activation(activation="gelu")(K)
        K = BatchNormalization()(K)

        Q = Conv2D(filters=k_channels,
                   kernel_size=1,
                   strides=1,
                   padding="valid")(input_tensor)
        Q = Activation(activation="gelu")(Q)
        Q = BatchNormalization()(Q)

        V = Conv2D(filters=v_channels,
                   kernel_size=1,
                   strides=1,
                   padding="valid")(input_tensor)
        V = Activation(activation="gelu")(V)
        V = BatchNormalization()(V)

        K = tf.stack([K[:, :, :w, :], K[:, :, w:, :]], axis=4)
        Q = tf.stack([Q[:, :, :w, :], Q[:, :, w:, :]], axis=4)
        V = tf.stack([V[:, :, :w, :], V[:, :, w:, :]], axis=4)

        def func(value_local, query_local, key_local):

            h_local, w_local = value_local.shape[1], value_local.shape[2]

            value_local = tf.reshape(value_local,
                                     shape=[-1, v_channels, h_local*w_local*2])
            query_local = tf.reshape(query_local,
                                     shape=[-1, k_channels, h_local*w_local*2])
            query_local = tf.transpose(query_local, perm=[0,2,1])
            key_local = tf.reshape(key_local,
                                   shape=[-1, k_channels, h_local*w_local*2])

            sim_map = tf.matmul(query_local, key_local)/tf.math.sqrt(tf.cast(k_channels, tf.float32))
            sim_map = tf.keras.activations.softmax(sim_map, axis=-1)
            context_local = tf.matmul(value_local, tf.transpose(sim_map, perm=[0, 2, 1]))
            context_local = tf.reshape(context_local,
                                       shape=[-1, h_local, w_local, v_channels, 2])
            return context_local

        local_block_cnt = scale*scale*2

        v_list = [V[:, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1], :] for i in
                  range(0, local_block_cnt, 2)]
        v_locals = tf.concat(v_list, axis=0)

        q_list = [Q[:, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1], :] for i in
                  range(0, local_block_cnt, 2)]
        q_locals = tf.concat(q_list, axis=0)

        k_list = [K[:, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1], :] for i in
                  range(0, local_block_cnt, 2)]
        k_locals = tf.concat(k_list, axis=0)

        context_locals = func(v_locals, q_locals, k_locals)
        context = tf.reshape(context_locals,
                             shape=[-1, h, 2*w, context_locals.shape[3]])
        if d_sample != 1 :
            context = UpSampling2D(size=(d_sample, d_sample),
                                   interpolation="bilinear")(context)

        #context = context+input_tensor
        context_list.append(context)

    context = Concatenate(axis=-1)(context_list)
    context = Conv2D(filters=c,#context.shape[-1],
                     kernel_size=1,
                     strides=1,
                     padding="valid")(context)
    context = Activation(activation="gelu")(context)
    context = BatchNormalization()(context)
    context = context + input_tensor
    return context

def contrastive_loss_reverse(y_true,
                             y_pred):
    """
    batch-balanced contrastive loss(BCL) version
    :param y_true:
    :param y_pred:
    :return:
    """
    # set margin
    margin = 2

    # the number of pixels in ground truth for each class (binary)
    num_of_change = tf.reduce_sum(y_true)+K.epsilon()
    num_of_nochange = tf.reduce_sum(1-y_true)+K.epsilon()

    # change
    change = tf.reduce_sum(y_true*tf.pow(tf.maximum(0., margin - y_pred),2))/num_of_change
    # no change parts
    no_change = tf.reduce_sum((1-y_true)*tf.pow(y_pred, 2))/num_of_nochange

    loss = change + no_change
    return loss

"""---BiT--- : 21.06.21. ~ """

"""---SE module--- : 21.07.26. ~ """
def se_module(input_feature,
              r=16):
    """

    :param input_feature:
    :param bottle_neck:
    :return:
    """
    feature_shape = input_feature.shape
    bottle_neck_ratio = int(feature_shape[-1]/r)

    squeeze = GlobalAvgPool2D(input_feature)
    FCL1 = Dense(units=bottle_neck_ratio)(squeeze)
    non_lin1 = Activation(activation="relu")(FCL1)

    FCL2 = Dense(units=feature_shape[-1])(non_lin1)
    non_lin2 = Activation(activation="sigmoid")(FCL2)

    return non_lin2

if __name__ == "__main__" :

    folder_dir = "E:\\Data_list\\2020_Landcover\\Data\\mask_mod"
    file_list = os.listdir(folder_dir)
    for file in file_list :
        file_dir = os.path.join(folder_dir, file)
        img = cv2.imread(file_dir, cv2.IMREAD_UNCHANGED)
        class2color(img, file)


