import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, AvgPool2D, MaxPool2D, Activation
from tensorflow.keras.layers import Add, Concatenate, BatchNormalization
from tensorflow.keras import backend as K

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

def ResNet50(input_tensor) :

    # FRONT
    front_conv = Conv2D(filters=64,
                        kernel_size=7,
                        strides=2,
                        padding="same",
                        activation="relu",
                        name="Conv")(input_tensor)
    Pool = MaxPool2D(pool_size=3,
                     strides=2,
                     padding="same")(front_conv)
    # BODY
    RB1 = Resblock_bn(input=Pool, knum_in=64, knum_out=256, layer_name="RB1")
    RB2 = Resblock_bn(input=RB1, knum_in=64, knum_out=256, layer_name="RB2")
    RB3 = Resblock_bn(input=RB2, knum_in=64, knum_out=256, layer_name="RB3")
    RB4 = Resblock_bn(input=RB3, knum_in=128, knum_out=512, layer_name="RB4", verbose=True)
    RB5 = Resblock_bn(input=RB4, knum_in=128, knum_out=512, layer_name="RB5")
    RB6 = Resblock_bn(input=RB5, knum_in=128, knum_out=512, layer_name="RB6")
    RB7 = Resblock_bn(input=RB6, knum_in=128, knum_out=512, layer_name="RB7")
    RB8 = Resblock_bn(input=RB7, knum_in=256, knum_out=1024, layer_name="RB8", verbose=True)
    RB9 = Resblock_bn(input=RB8, knum_in=256, knum_out=1024, layer_name="RB9")
    RB10 = Resblock_bn(input=RB9, knum_in=256, knum_out=1024, layer_name="RB10")
    RB11 = Resblock_bn(input=RB10,knum_in=256, knum_out=1024, layer_name="RB11")
    RB12 = Resblock_bn(input=RB11, knum_in=256, knum_out=1024, layer_name="RB12")
    RB13 = Resblock_bn(input=RB12, knum_in=256, knum_out=1024, layer_name="RB13")
    RB14 = Resblock_bn(input=RB13, knum_in=512, knum_out=2048, layer_name="RB14", verbose=True)
    RB15 = Resblock_bn(input=RB14, knum_in=512, knum_out=2048, layer_name="RB15")
    RB16 = Resblock_bn(input=RB15, knum_in=512, knum_out=2048, layer_name="RB16")

    return RB16

def RPN(output_feature, num_anchors):

    # height, width, channel = output_feature.shape[1], output_feature.shape[2], output_feature.shape[3]

    f1 = Conv2D(filters=256,
                kernel_size=(3,3),
                padding="same",
                kernel_initializer="he_normal",
                name="RPN_L1")(output_feature)
    f1 = Activation(activation="relu")(f1)

    rpn_regr = Conv2D(filters=4*num_anchors,
                      kernel_size=(1,1),
                      padding="same",
                      kernel_initializer="uniform",
                      name="RPN_REGR")(f1)
    rpn_regr = Activation(activation="linear")(rpn_regr)

    rpn_cls = Conv2D(filters=num_anchors,
                     kernel_size=(1,1),
                     padding="same",
                     kernel_initializer="uniform",
                     name="RPN_CLS")(f1)
    rpn_cls = Activation(activation="sigmoid")(rpn_cls)

    return [rpn_cls, rpn_regr, f1]
