"""
21.02.08. Mon.
This code is Pyramid Attention Network(PAN) implemented by Kangmin Park, Lab for Sensor and Modeling,
referring "Pyramid Attention Network for Semantic Segmentation", 2018, Hanchao Li, etal.
"""
import os
os.path.join("..")
from backbones.utils import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Optionset
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
session = tf.compat.v1.InteractiveSession(config=config)

# Feature Pyramid Attention(FPA)
def FPA(input_tensor,
        name):

    tensor_shape = input_tensor.shape

    # GAP branch
    GAP = GlobalAvgPool2D(name=name+"_FPA_GAP")(input_tensor)
    # Conv2d expected 4dim, but GlobalAvgPool2D returns 2dim tensor. using tf.expand_dims twice, expand dims to 4dim.
    GAP = tf.expand_dims(tf.expand_dims(GAP, axis=1), axis=1)
    GAP_conv1 = Conv2D(filters=tensor_shape[3]/2,
                       kernel_size=1,
                       kernel_initializer="he_normal",
                       activation="relu",
                       name=name+"_GAP_conv1")(GAP)
    GAP_upsample = UpSampling2D(size=(tensor_shape[1], tensor_shape[2]),
                                interpolation="nearest",
                                name=name+"_GAP_upsample")(GAP_conv1)

    # Main stem
    main_conv1x1 = Conv2D(filters=tensor_shape[3]/2,
                          kernel_size=1,
                          kernel_initializer="he_normal",
                          activation="relu",
                          name=name+"_main_conv1")(input_tensor)


    # Pyramid Attention(Pixel-level attention)
    # 1/2
    conv7x7 = Conv2D(filters=tensor_shape[3],
                     kernel_size=3,
                     kernel_initializer="he_normal",
                     strides=2,
                     padding="same",
                     activation="relu",
                     name=name+"_main_conv7x7")(input_tensor)

    # 1/4
    conv5x5 = Conv2D(filters=tensor_shape[3],
                     kernel_size=3,
                     kernel_initializer="he_normal",
                     strides=2,
                     padding="same",
                     activation="relu",
                     name=name+"_main_conv5x5")(conv7x7)

    # 1/8
    conv3x3 = Conv2D(filters=tensor_shape[3],
                     kernel_size=3,
                     kernel_initializer="he_normal",
                     strides=2,
                     padding="same",
                     activation="relu",
                     name=name+"_main_conv3x3")(conv5x5)

    # 1/4
    conv_up1 = Conv2DTranspose(filters=tensor_shape[3]/2,
                               kernel_size=3,
                               kernel_initializer="he_normal",
                               strides=2,
                               padding="same",
                               activation="relu",
                               name=name+"_main_conv_up1")(conv3x3)

    conv5x5_2 = Conv2D(filters=tensor_shape[3]/2,
                       kernel_size=3,
                       kernel_initializer="he_normal",
                       padding="same",
                       activation="relu",
                       name=name+"_conv5x5_2")(conv5x5)
    feature_add1 = Add()([conv_up1, conv5x5_2])

    # 1/2
    conv_up2 = Conv2DTranspose(filters=tensor_shape[3]/2,
                               kernel_size=3,
                               kernel_initializer="he_normal",
                               strides=2,
                               padding="same",
                               activation="relu",
                               name=name+"_main_conv_up2")(feature_add1)
    conv7x7_2 = Conv2D(filters=tensor_shape[3]/2,
                       kernel_size=3,
                       kernel_initializer="he_normal",
                       padding="same",
                       activation="relu",
                       name=name+"_conv7x7_2")(conv7x7)
    feature_add2 = Add()([conv_up2, conv7x7_2])

    # original size
    conv_up3 = Conv2DTranspose(filters=tensor_shape[3]/2,
                               kernel_size=3,
                               kernel_initializer="he_normal",
                               strides=2,
                               padding="same",
                               activation="relu",
                               name=name+"_conv_up3")(feature_add2)

    # Attention
    attention = Multiply()([main_conv1x1, conv_up3]) + GAP_upsample

    return attention

# Global Attention Upsample(GAU)
def GAU(high_level_feature,
        low_level_feature,
        name,
        option=False):
    """

    :param high_level_feature: H x W x C/2
    :param low_level_feature: H x W x C
    :param name:
    :param option:
    :return:
    """

    # num of channel of high level feature < low level feature channels
    tensor_shape = high_level_feature.shape

    # high level feature processing
    GAP = GlobalAvgPool2D(name=name+"_GAU_GAP")(high_level_feature)
    GAP = tf.expand_dims(tf.expand_dims(GAP, axis=1), axis=1)
    GAP_conv1 = Conv2D(filters=tensor_shape[3],
                       kernel_size=1,
                       kernel_initializer="he_normal",
                       name=name+"_GAU_GAP_conv1")(GAP)
    GAP_batch = BatchNormalization()(GAP_conv1)
    GAP_relu = Activation(activation="relu")(GAP_batch)

    # low level feature processing
    conv3x3 = Conv2D(filters=tensor_shape[3],
                     kernel_size=3,
                     kernel_initializer="he_normal",
                     padding="same",
                     activation="relu",
                     name=name+"_GAU_conv3x3")(low_level_feature)

    # attenion
    GAU_attention = Multiply()([GAP_relu, conv3x3])

    return GAU_attention

class PAN :

    def __init__(self, INPUT_SHAPE, NUM_CLASSES, LR):
        self.input_shape = INPUT_SHAPE
        self.num_classes = NUM_CLASSES
        self.lr = LR

    def build_net(self):

        input = keras.Input(shape=self.input_shape)

        # Encoder -ResNet101 / change conv7x7 into conv3x3
        # 1/2
        conv1 = Conv2D(filters=64,
                       strides=2,
                       padding="same",
                       kernel_size=3,
                       kernel_initializer="he_normal",
                       activation="relu",
                       name="encoder_conv1")(input)
        # 1/4
        pool1 = MaxPool2D(pool_size=3,
                          strides=2,
                          padding="same")(conv1)
        resb1 = Resblock_bn(input=pool1,
                            knum_in=64,
                            knum_out=256,
                            layer_name="resb1")
        resb2 = Resblock_bn(input=resb1,
                            knum_in=64,
                            knum_out=256,
                            layer_name="resb2")
        resb3 = Resblock_bn(input=resb2,
                            knum_in=64,
                            knum_out=256,
                            layer_name="resb3")

        # 1/8
        resb4 = Resblock_bn(input=resb3,
                            knum_in=128,
                            knum_out=512,
                            verbose=True,
                            layer_name="resb4")
        resb5 = Resblock_bn(input=resb4,
                            knum_in=128,
                            knum_out=512,
                            layer_name="resb5")
        resb6 = Resblock_bn(input=resb5,
                            knum_in=128,
                            knum_out=512,
                            layer_name="resb6")
        resb7 = Resblock_bn(input=resb6,
                            knum_in=128,
                            knum_out=512,
                            layer_name="resb7")

        # 1/16
        resb8 = Resblock_bn(input=resb6,
                            knum_in=256,
                            knum_out=1024,
                            layer_name="resb8")

        for i in range(0,22):
            key = False
            if i == 0 : key = True
            resb8 = Resblock_bn(input=resb8,
                                knum_in=256,
                                knum_out=1024,
                                verbose=key,
                                layer_name="resb{0}".format(i+9))
        # dilation
        resb31 = Resblock_bn(input=resb8,
                             knum_in=512,
                             knum_out=2048,
                             dilation=(2,2),
                             layer_name="resb31")
        resb32 = Resblock_bn(input=resb31,
                             knum_in=512,
                             knum_out=2048,
                             dilation=(2, 2),
                             layer_name="resb32")
        resb33 = Resblock_bn(input=resb32,
                             knum_in=512,
                             knum_out=2048,
                             dilation=(2, 2),
                             layer_name="resb33")

        fpa = FPA(resb33,
                  name="fpa_module")

        gau_att1 = GAU(high_level_feature=fpa,
                       low_level_feature=resb8,
                       name="gau_att1")
        de_feature1 = Add()([gau_att1, fpa])

        # 1/8
        de_up1 = Conv2DTranspose(filters=resb7.shape[3]/2,
                                 kernel_size=3,
                                 strides=2,
                                 padding="same",
                                 activation="relu",
                                 name="de_up1")(de_feature1)
        gau_att2 = GAU(high_level_feature=de_up1,
                       low_level_feature=resb7,
                       name="gau_att2")
        de_feature2 = Add()([gau_att2, de_up1])

        # 1/4
        de_up2 = Conv2DTranspose(filters=resb3.shape[3]/2,
                                 kernel_size=3,
                                 strides=2,
                                 padding="same",
                                 activation="relu",
                                 name="de_up2")(de_feature2)
        gau_att3 = GAU(high_level_feature=de_up2,
                       low_level_feature=resb3,
                       name="gau_att3")
        de_feature3 = Add()([gau_att3, de_up2])

        # 1/2
        de_up3 = Conv2DTranspose(filters=de_up2.shape[3]/2,
                                 kernel_size=3,
                                 strides=2,
                                 padding="same",
                                 activation="relu",
                                 name="de_up3")(de_feature3)
        de_up4 = Conv2DTranspose(filters=de_up3.shape[3]/2,
                                 kernel_size=3,
                                 strides=2,
                                 padding="same",
                                 activation="relu",
                                 name="de_up4")(de_up3)
        output = Conv2D(filters=self.num_classes,
                        kernel_size=3,
                        padding="same",
                        activation="softmax",
                        name="output")(de_up4)

        model = keras.Model(inputs=input, outputs=output)

        return model

if __name__ == "__main__" :

    INPUT_SHAPE = (512, 512, 3)
    NUM_CLS = 2
    LEARNING_RATE = 1e-4

    PAN = PAN(INPUT_SHAPE=INPUT_SHAPE, NUM_CLASSES=NUM_CLS, LR=LEARNING_RATE)
    model = PAN.build_net()
    model.summary()