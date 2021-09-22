import sys
import cv2
sys.path.append("./backbones")
import numpy as np
from utils import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.utils import losses_utils
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPool2D
from tensorflow.keras.layers import BatchNormalization, Concatenate, Add, Activation
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy, BinaryCrossentropy, MeanSquaredError, categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.keras.applications import ResNet50, ResNet101
from tensorflow.keras.metrics import MeanIoU

# Optionset
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
session = tf.compat.v1.InteractiveSession(config=config)

class UpdatedMeanIoU(keras.metrics.MeanIoU) :
    def __init__(self,
                 y_true = None,
                 y_pred = None,
                 num_classes=None,
                 name=None,
                 dtype=None):
        super(UpdatedMeanIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

def hierachical_weight(y_pred, y_true, weight_table):

    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    y_true = tf.argmax(y_true, axis=-1)

    weight_table = tf.convert_to_tensor(weight_table)
    index_tensor = tf.stack([y_pred, y_true], axis=-1)
    weights_for_loss = tf.gather_nd(weight_table, index_tensor)

    return tf.cast(weights_for_loss, tf.float32)

def hierachical_weight_2(y_pred, y_true) :

    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    weights = tf.math.pow(y_pred-y_true, 2)
    return weights

def class_weight_matrix(y_true, class_weight):
    """
    class weights for class balancing.
    :param y_true: multi channel matrix format ( not semantic label format )
    :param class_weight: dictionary type? list type?
    :return:
    """
    y_true = tf.argmax(y_true, axis=-1)
    y_true_index = K.expand_dims(y_true, axis=-1)
    class_weight_tensor = tf.convert_to_tensor(class_weight)
    class_weight = tf.gather_nd(class_weight_tensor, y_true_index)

    return tf.cast(class_weight, tf.float32)

def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1):
    """Computes the sparse categorical crossentropy loss.

    Usage:

    y_true = [1, 2]
    y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    assert loss.shape == (2,)
    loss.numpy()
    array([0.0513, 2.303], dtype=float32)

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.
      from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
        we assume that `y_pred` encodes a probability distribution.
      axis: (Optional) Defaults to -1. The dimension along which the entropy is
        computed.

    Returns:
      Sparse categorical crossentropy loss value.
    """
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    # Modified parts.
    # weight_table = hierachical_weight_table(42, 1, 2, 4, 8)
    # weight = hierachical_weight(y_pred, y_true, weight_table)
    # print(weight.shape)
    class_weight = [0.1, 0.2, 0.3, 0.4, 0.5,
                    0.1, 0.2, 0.3, 0.4, 0.5,
                    0.1, 0.2, 0.3, 0.4, 0.5,
                    0.1, 0.2, 0.3, 0.4, 0.5,
                    0.1, 0.2, 0.3, 0.4, 0.5,
                    0.1, 0.2, 0.3, 0.4, 0.5,
                    0.1, 0.2, 0.3, 0.4, 0.5,
                    0.1, 0.2, 0.3, 0.4, 0.5,
                    0.1, 0.2]
    weight = class_weight_matrix(y_true, class_weight)
    print(weight.shape)

    return K.sparse_categorical_crossentropy(
       y_true, y_pred, from_logits=from_logits, axis=axis)*weight

class UpdatedSparseCategoricalCrossentropy(SparseCategoricalCrossentropy):

    def __init__(self,
                 from_logits=False,
                 reduction=losses_utils.ReductionV2.NONE,
                 name='sparse_categorical_crossentropy'):
        super(SparseCategoricalCrossentropy, self).__init__(
            sparse_categorical_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits)

class DlinkNet34 :

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.build_net()

    def build_net(self):

        input = keras.Input(self.input_shape)

        # Encoder : ResNet34
        # HEAD
        Conv = Conv2D(filters=64, kernel_size=7, kernel_initializer="he_normal",
                      strides=2, activation="relu", padding="same", name="Conv")(input)
        Pool = MaxPool2D(pool_size=3, strides=2, padding="same")(Conv)

        # BODY
        RB1 = Resblock(Pool, knum=64, layer_name="RB1")
        RB2 = Resblock(RB1, knum=64, layer_name="RB2")
        RB3 = Resblock(RB2, knum=64, layer_name="RB3")

        RB4 = Resblock(RB3, knum=128, layer_name="RB4", verbose=True)
        RB5 = Resblock(RB4, knum=128, layer_name="RB5")
        RB6 = Resblock(RB5, knum=128, layer_name="RB6")
        RB7 = Resblock(RB6, knum=128, layer_name="RB7")

        RB8 = Resblock(RB7, knum=256, layer_name="RB8", verbose=True)
        RB9 = Resblock(RB8, knum=256, layer_name="RB9")
        RB10 = Resblock(RB9, knum=256, layer_name="RB10")
        RB11 = Resblock(RB10, knum=256, layer_name="RB11")
        RB12 = Resblock(RB11, knum=256, layer_name="RB12")
        RB13 = Resblock(RB12, knum=256, layer_name="RB13")

        RB14 = Resblock(RB13, knum=512, layer_name="RB14", verbose=True)
        RB15 = Resblock(RB14, knum=512, layer_name="RB15")
        RB16 = Resblock(RB15, knum=512, layer_name="RB16")

        # Dilated layers
        D_block = Dilated_layers(RB16, knum=512, ksize=3, layer_name="D_block")

        # Transposed Convolution layers
        TB1 = Transblock(input=D_block, knum_out=256, layer_name="TB1")
        TB1_shortcut = Add()([RB13, TB1])

        TB2 = Transblock(input=TB1_shortcut, knum_out=128, layer_name="TB2")
        TB2_shortcut = Add()([RB7, TB2])

        TB3 = Transblock(input=TB2_shortcut, knum_out=64, layer_name="TB3")
        TB3_shortcut = Add()([RB3, TB3])

        TB4 = Transblock(input=TB3_shortcut, knum_out=64, layer_name="TB4")
        Trans_conv = Conv2DTranspose(filters=32, kernel_size=4, kernel_initializer="he_normal",
                                     activation="relu", strides=2, padding="same", name="tail_trans")(TB4)
        output = Conv2D(filters=self.num_classes, kernel_size=3,
                        strides=1, padding="same", name="output", activation="softmax")(Trans_conv)
        model = keras.Model(inputs=input, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=UpdatedSparseCategoricalCrossentropy(),
            #metrics=keras.metrics.MeanIoU(num_classes=self.num_classes)
            metrics=UpdatedMeanIoU(num_classes=self.num_classes)
            #metrics=["accuracy"]
        )

        return model

class DlinkNet101:

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.build_net()

    def build_net(self):

        input = keras.Input(self.input_shape)

        #Head
        Conv = Conv2D(filters=64, kernel_size=7, kernel_initializer="he_normal",
                      strides=2, padding="same", activation="relu", name="Conv")(input)
        Maxp = MaxPool2D(pool_size=3, strides=2, padding="same")(Conv)

        #BODY
        RB1 = Resblock_bn(input=Maxp, knum_in=64, knum_out=256, layer_name="RB1")
        RB2 = Resblock_bn(input=RB1, knum_in=64, knum_out=256, layer_name="RB2")
        RB3 = Resblock_bn(input=RB2, knum_in=64, knum_out=256, layer_name="RB3")

        RB4 = Resblock_bn(input=RB3, knum_in=128, knum_out=512, layer_name="RB4", verbose=True)
        RB5 = Resblock_bn(input=RB4, knum_in=128, knum_out=512, layer_name="RB5")
        RB6 = Resblock_bn(input=RB5, knum_in=128, knum_out=512, layer_name="RB6")
        RB7 = Resblock_bn(input=RB6, knum_in=128, knum_out=512, layer_name="RB7")

        RB8 = Resblock_bn(input=RB7, knum_in=256, knum_out=1024, layer_name="RB8", verbose=True)
        RB9 = Resblock_bn(input=RB8, knum_in=256, knum_out=1024, layer_name="RB9")
        RB10 = Resblock_bn(input=RB9, knum_in=256, knum_out=1024, layer_name="RB10")
        RB11 = Resblock_bn(input=RB10, knum_in=256, knum_out=1024, layer_name="RB11")
        RB12 = Resblock_bn(input=RB11, knum_in=256, knum_out=1024, layer_name="RB12")
        RB13 = Resblock_bn(input=RB12, knum_in=256, knum_out=1024, layer_name="RB13")
        RB14 = Resblock_bn(input=RB13, knum_in=256, knum_out=1024, layer_name="RB14")
        RB15 = Resblock_bn(input=RB14, knum_in=256, knum_out=1024, layer_name="RB15")
        RB16 = Resblock_bn(input=RB15, knum_in=256, knum_out=1024, layer_name="RB16")
        RB17 = Resblock_bn(input=RB16, knum_in=256, knum_out=1024, layer_name="RB17")
        RB18 = Resblock_bn(input=RB17, knum_in=256, knum_out=1024, layer_name="RB18")
        RB19 = Resblock_bn(input=RB18, knum_in=256, knum_out=1024, layer_name="RB19")
        RB20 = Resblock_bn(input=RB19, knum_in=256, knum_out=1024, layer_name="RB20")
        RB21 = Resblock_bn(input=RB20, knum_in=256, knum_out=1024, layer_name="RB21")
        RB22 = Resblock_bn(input=RB21, knum_in=256, knum_out=1024, layer_name="RB22")
        RB23 = Resblock_bn(input=RB22, knum_in=256, knum_out=1024, layer_name="RB23")
        RB24 = Resblock_bn(input=RB23, knum_in=256, knum_out=1024, layer_name="RB24")
        RB25 = Resblock_bn(input=RB24, knum_in=256, knum_out=1024, layer_name="RB25")
        RB26 = Resblock_bn(input=RB25, knum_in=256, knum_out=1024, layer_name="RB26")
        RB27 = Resblock_bn(input=RB26, knum_in=256, knum_out=1024, layer_name="RB27")
        RB28 = Resblock_bn(input=RB27, knum_in=256, knum_out=1024, layer_name="RB28")
        RB29 = Resblock_bn(input=RB28, knum_in=256, knum_out=1024, layer_name="RB29")
        RB30 = Resblock_bn(input=RB29, knum_in=256, knum_out=1024, layer_name="RB30")

        RB31 = Resblock_bn(input=RB30, knum_in=512, knum_out=2048, layer_name="RB31", verbose=True)
        RB32 = Resblock_bn(input=RB31, knum_in=512, knum_out=2048, layer_name="RB32")
        RB33 = Resblock_bn(input=RB32, knum_in=512, knum_out=2048, layer_name="RB33")

        D_block = Dilated_layers(input=RB33, knum=2048, ksize=3, layer_name="D_block")

        TB1 = Transblock(input=D_block, knum_out=1024, layer_name="TB1")
        TB1_shortcut = Add()([RB30, TB1])

        TB2 = Transblock(input=TB1_shortcut, knum_out=512, layer_name="TB2")
        TB2_shortcut = Add()([TB2, RB7])

        TB3 = Transblock(input=TB2_shortcut, knum_out=256, layer_name="TB3")
        TB3_shortcut = Add()([TB3, RB3])

        TB4 = Transblock(input=TB3_shortcut, knum_out=128, layer_name="TB4")
        Trans_conv = Conv2DTranspose(filters=128, kernel_size=4, kernel_initializer="he_normal",
                                     activation="relu", strides=2, padding="same", name="Trans_conv")(TB4)
        TAIL = Conv2D(filters=64, kernel_size=3, kernel_initializer="he_normal",
                      activation="relu", strides=1, padding="same", name="TAIL_conv")(Trans_conv)
        output = Conv2D(filters=self.num_classes, kernel_size=3,
                        activation="softmax", strides=1, padding="same", name="TAIL_conv2")(TAIL)
        model = keras.Model(inputs=input, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss = SparseCategoricalCrossentropy(),
            metrics= UpdatedMeanIoU(num_classes=self.num_classes)
        )

        return model

class DlinkNet101_imagenet:

    def __init__(self, input_shape, num_classes, imagenet_trainable):
        self.input_shape=input_shape
        self.num_classes=num_classes
        self.imagenet_trainable=imagenet_trainable
        self.build_net()

    def build_net(self):

        input = keras.Input(self.input_shape)
        Resnet_imagenet = ResNet101(input_tensor=input, include_top=False, weights="imagenet")
        Resnet_imagenet.trainable=self.imagenet_trainable
        Resnet_output = Resnet_imagenet.output

        D_block = Dilated_layers(input=Resnet_output, knum=2048, ksize=3, layer_name="D_block")

        TB1 = Transblock(input=D_block, knum_out=1024, layer_name="TB1")
        RB30 = Resnet_imagenet.get_layer(name="conv4_block23_out").output
        TB1_shortcut = Add()([TB1, RB30])

        TB2 = Transblock(input=TB1_shortcut, knum_out=512, layer_name="TB2")
        RB7 = Resnet_imagenet.get_layer(name="conv3_block4_out").output
        TB2_shortcut = Add()([TB2, RB7])

        TB3 = Transblock(input=TB2_shortcut, knum_out=256, layer_name="TB3")
        RB3 = Resnet_imagenet.get_layer(name="conv2_block3_out").output
        TB3_shortcut = Add()([TB3, RB3])

        TB4 = Transblock(input=TB3_shortcut, knum_out=128, layer_name="TB4")
        Trans_conv = Conv2DTranspose(filters=128, kernel_size=4, kernel_initializer="he_normal",
                                     activation="relu", strides=2, padding="same", name="Trans_conv")(TB4)
        TAIL = Conv2D(filters=64, kernel_size=3, kernel_initializer="he_normal",
                      activation="relu", strides=1, padding="same", name="TAIL_conv")(Trans_conv)
        output = Conv2D(filters=self.num_classes, kernel_size=3,
                        activation="softmax", strides=1, padding="same", name="TAIL_conv2")(TAIL)
        model = keras.Model(inputs=input, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(0.005),
            loss=SparseCategoricalCrossentropy(),
            metrics=UpdatedMeanIoU(num_classes=self.num_classes)
        )

        return model

class DlinkNet50_h :

    def __init__(self, input_shape, num_classes, imagenet_trainable, lr):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.imagenet_trainable = imagenet_trainable
        self.lr = lr
        self.build_net()

    def build_net(self):

        input = keras.Input(self.input_shape)
        Resnet_imagenet = ResNet50(input_tensor=input, include_top=False, weights="imagenet")
        Resnet_imagenet.trainable = self.imagenet_trainable
        Resnet_output = Resnet_imagenet.output

        D_block = Dilated_layers(input=Resnet_output, knum=2048, ksize=3, layer_name="D_block")

        TB1 = Transblock(input=D_block, knum_out=1024, layer_name="TB1")
        RB30 = Resnet_imagenet.get_layer(name="conv4_block6_out").output
        TB1_shortcut = Add()([TB1, RB30])

        # Main category loss function
        aux1_transcov1 = Conv2DTranspose(filters=512, kernel_size=4, kernel_initializer="he_normal",
                                         activation="relu", strides=4, padding="same", name="aux1_transcov1")(
            TB1_shortcut)
        aux1_transconv2 = Conv2DTranspose(filters=256, kernel_size=4, kernel_initializer="he_normal",
                                          activation="relu", strides=4, padding="same", name="aux1_transcov2")(
            aux1_transcov1)
        aux1_output = Conv2D(filters=self.num_classes, kernel_size=1, kernel_initializer="he_normal",
                             activation="softmax", strides=1, padding="same", name="aux1_output")(aux1_transconv2)

        TB2 = Transblock(input=TB1_shortcut, knum_out=512, layer_name="TB2")
        RB7 = Resnet_imagenet.get_layer(name="conv3_block4_out").output
        TB2_shortcut = Add()([TB2, RB7])

        # Middle category loss function
        aux2_transcov1 = Conv2DTranspose(filters=256, kernel_size=4, kernel_initializer="he_normal",
                                         activation="relu", strides=4, padding="same", name="aux2_transcov1")(
            TB2_shortcut)
        aux2_transconv2 = Conv2DTranspose(filters=128, kernel_size=3, kernel_initializer="he_normal",
                                          activation="relu", strides=2, padding="same", name="aux2_transcov2")(
            aux2_transcov1)
        aux2_output = Conv2D(filters=self.num_classes, kernel_size=1, kernel_initializer="he_normal",
                             activation="softmax", strides=1, padding="same", name="aux2_output")(aux2_transconv2)

        TB3 = Transblock(input=TB2_shortcut, knum_out=256, layer_name="TB3")
        RB3 = Resnet_imagenet.get_layer(name="conv2_block3_out").output
        TB3_shortcut = Add()([TB3, RB3])

        TB4 = Transblock(input=TB3_shortcut, knum_out=128, layer_name="TB4")
        Trans_conv = Conv2DTranspose(filters=128, kernel_size=4, kernel_initializer="he_normal",
                                     activation="relu", strides=2, padding="same", name="Trans_conv")(TB4)
        TAIL = Conv2D(filters=64, kernel_size=3, kernel_initializer="he_normal",
                      activation="relu", strides=1, padding="same", name="TAIL_conv")(Trans_conv)
        output = Conv2D(filters=self.num_classes, kernel_size=3,
                        activation="softmax", strides=1, padding="same", name="TAIL_conv2")(TAIL)
        model = keras.Model(inputs=input, outputs=[output, aux1_output, aux2_output])
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=[SparseCategoricalCrossentropy(), SparseCategoricalCrossentropy(), SparseCategoricalCrossentropy()],
            metrics=UpdatedMeanIoU(num_classes=self.num_classes)
        )

        return model

class DlinkNet50_detail :

    def __init__(self, input_shape, num_classes, resnet_trainable, lr):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.resnet_trainable = resnet_trainable
        self.lr = lr

    def build_net(self):

        input = keras.Input(self.input_shape)
        resnet = ResNet50(input_tensor=input, include_top=False, weights="imagenet")
        resnet.trainable = self.resnet_trainable
        resnet_output = resnet.output

        D_block = Dilated_layers(input=resnet_output, knum=2048, ksize=3, layer_name="D_block")

        TB1 = Transblock(input=D_block, knum_out=1024, layer_name="TB1")
        RB3 = resnet.get_layer(name="conv4_block6_out").output
        TB1_shortcut = Add()([TB1, RB3])

        TB2 = Transblock(input=TB1_shortcut, knum_out=512, layer_name="TB2")
        RB2 = resnet.get_layer(name="conv3_block4_out").output
        TB2_shortcut = Add()([TB2, RB2])

        TB3 = Transblock(input=TB2_shortcut, knum_out=256, layer_name="TB3")
        RB1 = resnet.get_layer(name="conv2_block3_out").output
        TB3_shortcut = Add()([TB3, RB1])

        TB4 = Transblock(input=TB3_shortcut, knum_out=128, layer_name="TB4")
        Trans_conv = Conv2DTranspose(filters=128, kernel_size=4, kernel_initializer="he_normal",
                                     activation="relu", strides=2, padding="same", name="Trans_conv")(TB4)
        TAIL = Conv2D(filters=64, kernel_size=3, kernel_initializer="he_normal",
                      activation="relu", strides=1, padding="same", name="TAIL_conv")(Trans_conv)
        output = Conv2D(filters=self.num_classes, kernel_size=3,
                        activation="softmax", strides=1, padding="same", name="TAIL_conv2")(TAIL)
        model = keras.Model(inputs=input, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=UpdatedSparseCategoricalCrossentropy(),
            metrics=UpdatedMeanIoU(num_classes=self.num_classes)
        )

        return model

class DlinkNet101_h :

    def __init__(self, input_shape, num_classes1, num_classes2, num_classes3, imagenet_trainable, lr):
        self.input_shape = input_shape
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.num_classes3 = num_classes3
        self.imagenet_trainable = imagenet_trainable
        self.lr = lr
        self.build_net()

    def build_net(self):

        input = keras.Input(self.input_shape)
        Resnet_imagenet = ResNet101(input_tensor=input, include_top=False, weights="imagenet")
        Resnet_imagenet.trainable = self.imagenet_trainable
        Resnet_output = Resnet_imagenet.output

        D_block = Dilated_layers(input=Resnet_output, knum=2048, ksize=3, layer_name="D_block")

        TB1 = Transblock(input=D_block, knum_out=1024, layer_name="TB1")
        RB30 = Resnet_imagenet.get_layer(name="conv4_block23_out").output
        TB1_shortcut = Add()([TB1, RB30])

        # Main category loss function
        aux1_transcov1 = Conv2DTranspose(filters=512, kernel_size=4, kernel_initializer="he_normal",
                                         activation="relu", strides=4, padding="same", name="aux1_transcov1")(TB1_shortcut)
        aux1_transconv2 = Conv2DTranspose(filters=256, kernel_size=4, kernel_initializer="he_normal",
                                         activation="relu", strides=4, padding="same", name="aux1_transcov2")(aux1_transcov1)
        aux1_output = Conv2D(filters=self.num_classes1, kernel_size=1, kernel_initializer="he_normal",
                             activation="softmax", strides=1, padding="same", name="aux1_output")(aux1_transconv2)

        TB2 = Transblock(input=TB1_shortcut, knum_out=512, layer_name="TB2")
        RB7 = Resnet_imagenet.get_layer(name="conv3_block4_out").output
        TB2_shortcut = Add()([TB2, RB7])

        # Middle category loss function
        aux2_transcov1 = Conv2DTranspose(filters=256, kernel_size=4, kernel_initializer="he_normal",
                                         activation="relu", strides=4, padding="same", name="aux2_transcov1")(TB2_shortcut)
        aux2_transconv2 = Conv2DTranspose(filters=128, kernel_size=3, kernel_initializer="he_normal",
                                          activation="relu", strides=2, padding="same", name="aux2_transcov2")(aux2_transcov1)
        aux2_output = Conv2D(filters=self.num_classes2, kernel_size=1, kernel_initializer="he_normal",
                             activation="softmax", strides=1, padding="same", name="aux2_output")(aux2_transconv2)

        TB3 = Transblock(input=TB2_shortcut, knum_out=256, layer_name="TB3")
        RB3 = Resnet_imagenet.get_layer(name="conv2_block3_out").output
        TB3_shortcut = Add()([TB3, RB3])

        # Detail category loss function
        TB4 = Transblock(input=TB3_shortcut, knum_out=128, layer_name="TB4")
        Trans_conv = Conv2DTranspose(filters=128, kernel_size=4, kernel_initializer="he_normal",
                                     activation="relu", strides=2, padding="same", name="Trans_conv")(TB4)
        TAIL = Conv2D(filters=64, kernel_size=3, kernel_initializer="he_normal",
                      activation="relu", strides=1, padding="same", name="TAIL_conv")(Trans_conv)
        output = Conv2D(filters=self.num_classes3, kernel_size=3,
                        activation="softmax", strides=1, padding="same", name="TAIL_conv2")(TAIL)
        model = keras.Model(inputs=input, outputs=[output, aux1_output, aux2_output])
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=[SparseCategoricalCrossentropy(), SparseCategoricalCrossentropy(), SparseCategoricalCrossentropy()],
            metrics=UpdatedMeanIoU(num_classes=self.num_classes3)
        )

        return model

class DlinkNet101_3tails :

    def __init__(self, input_shape, num_classes1, num_classes2, num_classes3, imagenet_trainable, lr):
        self.input_shape = input_shape
        # Main
        self.num_classes1 = num_classes1
        # Middle
        self.num_classes2 = num_classes2
        # Detail
        self.num_classes3 = num_classes3
        self.imagenet_trainable = imagenet_trainable
        self.lr = lr
        self.build_net()

    def build_net(self):

        input = keras.Input(self.input_shape)
        Resnet_imagenet = ResNet101(input_tensor=input, include_top=False, weights="imagenet")
        Resnet_imagenet.trainable = self.imagenet_trainable
        Resnet_output = Resnet_imagenet.output

        D_block = Dilated_layers(input=Resnet_output, knum=2048, ksize=3, layer_name="D_block")

        # TAIL1
        TAIL1_TB1 = Transblock(input=D_block, knum_out=1024, layer_name="TAIL1_TB1")
        TAIL1_RB30 = Resnet_imagenet.get_layer(name="conv4_block23_out").output
        TAIL1_TB1_shortcut = Add()([TAIL1_TB1, TAIL1_RB30])

        TAIL1_TB2 = Transblock(input=TAIL1_TB1_shortcut, knum_out=512, layer_name="TAIL1_TB2")
        TAIL1_RB7 = Resnet_imagenet.get_layer(name="conv3_block4_out").output
        TAIL1_TB2_shortcut = Add()([TAIL1_TB2, TAIL1_RB7])

        TAIL1_TB3 = Transblock(input=TAIL1_TB2_shortcut, knum_out=256, layer_name="TAIL1_TB3")
        TAIL1_RB3 = Resnet_imagenet.get_layer(name="conv2_block3_out").output
        TAIL1_TB3_shortcut = Add()([TAIL1_TB3, TAIL1_RB3])

        TAIL1_TB4 = Transblock(input=TAIL1_TB3_shortcut, knum_out=128, layer_name="TAIL1_TB4")
        TAIL1_Trans_conv = Conv2DTranspose(filters=128, kernel_size=4, kernel_initializer="he_normal",
                                     activation="relu", strides=2, padding="same", name="TAIL1_Trans_conv")(TAIL1_TB4)
        TAIL1 = Conv2D(filters=32, kernel_size=3, kernel_initializer="he_normal",
                      activation="relu", strides=1, padding="same", name="TAIL1_conv")(TAIL1_Trans_conv)
        output1 = Conv2D(filters=self.num_classes1, kernel_size=3,
                        activation="softmax", strides=1, padding="same", name="output1")(TAIL1)

        hier_weight1 = hierachical_weights(tensor=output1, category="main2mid")

        #TAIL2
        TAIL2_TB1 = Transblock(input=D_block, knum_out=512, layer_name="TAIL2_TB1")
        TAIL1_TB1_shortcut_reduce = Conv2D(filters=512, kernel_size=1, kernel_initializer="he_normal",
                                           activation="relu", strides=1, padding="same", name="TAIL1_TB1_shortcut_reduce")(TAIL1_TB1_shortcut)
        TAIL2_TB1_shortcut = Concatenate()([TAIL2_TB1, TAIL1_TB1_shortcut_reduce])

        TAIL2_TB2 = Transblock(input=TAIL2_TB1_shortcut, knum_out=256, layer_name="TAIL2_TB2")
        TAIL1_TB2_shortcut_reduce = Conv2D(filters=256, kernel_size=1, kernel_initializer="he_normal",
                                           activation="relu", strides=1, padding="same", name="TAIL1_TB2_shortcut_reduce")(TAIL1_TB2_shortcut)
        TAIL2_TB2_shortcut = Concatenate()([TAIL2_TB2, TAIL1_TB2_shortcut_reduce])

        TAIL2_TB3 = Transblock(input=TAIL2_TB2_shortcut, knum_out=128, layer_name="TAIL2_TB3")
        TAIL1_TB3_shortcut_reduce = Conv2D(filters=128, kernel_size=1, kernel_initializer="he_normal",
                                           activation="relu", strides=1, padding="same", name="TAIL1_TB3_shortcut_reduce")(TAIL1_TB3_shortcut)
        TAIL2_TB3_shortcut = Concatenate()([TAIL2_TB3, TAIL1_TB3_shortcut_reduce])

        TAIL2_TB4 = Transblock(input=TAIL2_TB3_shortcut, knum_out=128, layer_name="TAIL2_TB4")
        TAIL2_Trans_conv = Conv2DTranspose(filters=128, kernel_size=4, kernel_initializer="he_normal",
                                           activation="relu", strides=2, padding="same", name="TAIL2_Trans_conv")(TAIL2_TB4)
        TAIL2 = Conv2D(filters=64, kernel_size=3, kernel_initializer="he_normal",
                       activation="relu", strides=1, padding="same", name="TAIL2_conv")(TAIL2_Trans_conv)
        TAIL2 = Conv2D(filters=self.num_classes2, kernel_size=3,
                         strides=1, padding="same", name="TAIL2_conv2")(TAIL2)
        output2 = Add()([TAIL2, hier_weight1])
        output2 = Activation(activation="softmax", name="output2")(output2)

        hier_weight2 = hierachical_weights(tensor=output2, category="mid2det")

        # TAIL3
        TAIL3_TB1 = Transblock(input=D_block, knum_out=512, layer_name="TAIL3_TB1")
        TAIL2_TB1_shortcut_reduce = Conv2D(filters=512, kernel_size=1, kernel_initializer="he_normal",
                                           activation="relu", strides=1, padding="same", name="TAIL2_TB1_shortcut_reduce")(TAIL2_TB1_shortcut)
        TAIL3_TB1_shortcut = Concatenate()([TAIL3_TB1, TAIL2_TB1_shortcut_reduce])

        TAIL3_TB2 = Transblock(input=TAIL3_TB1_shortcut, knum_out=256, layer_name="TAIL3_TB2")
        TAIL2_TB2_shortcut_reduce = Conv2D(filters=256, kernel_size=1, kernel_initializer="he_normal",
                                           activation="relu", strides=1, padding="same", name="TAIL2_TB2_shortcut_reduce")(TAIL2_TB2_shortcut)
        TAIL3_TB2_shortcut = Concatenate()([TAIL3_TB2, TAIL2_TB2_shortcut_reduce])

        TAIL3_TB3 = Transblock(input=TAIL3_TB2_shortcut, knum_out=128, layer_name="TAIL3_TB3")
        TAIL2_TB3_shortcut_reduce = Conv2D(filters=128, kernel_size=1, kernel_initializer="he_normal",
                                           activation="relu", strides=1, padding="same", name="TAIL2_TB3_shortcut_reduce")(TAIL2_TB3_shortcut)
        TAIL3_TB3_shortcut = Concatenate()([TAIL3_TB3, TAIL2_TB3_shortcut_reduce])

        TAIL2_TB4 = Transblock(input=TAIL3_TB3_shortcut, knum_out=128, layer_name="TAIL3_TB4")
        TAIL3_Trans_conv = Conv2DTranspose(filters=128, kernel_size=4, kernel_initializer="he_normal",
                                           activation="relu", strides=2, padding="same", name="TAIL3_Trans_conv")(TAIL2_TB4)
        TAIL3 = Conv2D(filters=64, kernel_size=3, kernel_initializer="he_normal",
                       activation="relu", strides=1, padding="same", name="TAIL3_conv")(TAIL3_Trans_conv)
        TAIL3 = Conv2D(filters=self.num_classes3, kernel_size=3,
                         activation="softmax", strides=1, padding="same", name="TAIL3_conv2")(TAIL3)
        output3 = Add()([TAIL3, hier_weight2])
        output3 = Activation(activation="softmax", name="output3")(output3)

        model = keras.Model(inputs=input, outputs=[output1, output2, output3])
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss={"output1" : SparseCategoricalCrossentropy(), "output2" : SparseCategoricalCrossentropy(),"output3" : SparseCategoricalCrossentropy()},
            loss_weights={"output1" : 1, "output2": 1.5, "output3" : 3},
            metrics={"output1" : UpdatedMeanIoU(num_classes=self.num_classes1), "output2" : UpdatedMeanIoU(num_classes=self.num_classes2), "output3" : UpdatedMeanIoU(num_classes=self.num_classes3)}
        )

        return model

class DlinkNet101_main :

    def __init__(self, input_shape, num_classes1, imagenet_trainable, lr):
        self.input_shape = input_shape
        # Main
        self.num_classes1 = num_classes1
        self.imagenet_trainable = imagenet_trainable
        self.lr = lr
        self.build_net()

    def build_net(self):

        input = keras.Input(self.input_shape)
        Resnet_imagenet = ResNet101(input_tensor=input, include_top=False, weights="imagenet")
        Resnet_imagenet.trainable = self.imagenet_trainable
        Resnet_output = Resnet_imagenet.output

        D_block = Dilated_layers(input=Resnet_output, knum=2048, ksize=3, layer_name="Main_D_block")
	
        TB1 = Transblock(input=D_block, knum_out=1024, layer_name="TAIL1_TB1")
        RB30 = Resnet_imagenet.get_layer(name="conv4_block23_out").output
        TB1_shortcut = Add()([TB1, RB30])

        TB2 = Transblock(input=TB1_shortcut, knum_out=512, layer_name="TAIL1_TB2")
        RB7 = Resnet_imagenet.get_layer(name="conv3_block4_out").output
        TB2_shortcut = Add()([TB2, RB7])

        TB3 = Transblock(input=TB2_shortcut, knum_out=256, layer_name="TAIL1_TB3")
        RB3 = Resnet_imagenet.get_layer(name="conv2_block3_out").output
        TB3_shortcut = Add()([TB3, RB3])

        TB4 = Transblock(input=TB3_shortcut, knum_out=128, layer_name="TAIL1_TB4")
        Trans_conv = Conv2DTranspose(filters=128, kernel_size=4, kernel_initializer="he_normal",
                                     activation="relu", strides=2, padding="same", name="TAIL1_Trans_conv")(TB4)
        TAIL1 = Conv2D(filters=32, kernel_size=3, kernel_initializer="he_normal",
                      activation="relu", strides=1, padding="same", name="TAIL1_conv")(Trans_conv)
        output1 = Conv2D(filters=self.num_classes1, kernel_size=3,
                         activation="softmax", strides=1, padding="same", name="output1")(TAIL1)

        model = keras.Model(inputs=input, outputs=output1)
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=SparseCategoricalCrossentropy(),
            metrics=UpdatedMeanIoU(num_classes=self.num_classes1)
        )

        return model

class DlinkNet101_middle :

    def __init__(self, input_shape, num_classes2, imagenet_trainable, lr):
        self.input_shape = input_shape

        # Middle
        self.num_classes2 = num_classes2
        self.imagenet_trainable = imagenet_trainable
        self.lr = lr
        self.build_net()

    def build_net(self):

        input = keras.Input(self.input_shape)
        Resnet_imagenet = ResNet101(input_tensor=input, include_top=False, weights="imagenet")
        Resnet_imagenet.trainable = self.imagenet_trainable

        Resnet_output = Resnet_imagenet.output

        D_block = Dilated_layers(input=Resnet_output, knum=2048, ksize=3, layer_name="D_block")
	
	    # TAIL2
        TAIL2_TB1 = Transblock(input=D_block, knum_out=1024, layer_name="TAIL2_TB1")
        TAIL2_RB30 = Resnet_imagenet.get_layer(name="conv4_block23_out").output
        TAIL2_TB1_shortcut = Add()([TAIL2_TB1, TAIL2_RB30])

        TAIL2_TB2 = Transblock(input=TAIL2_TB1_shortcut, knum_out=512, layer_name="TAIL2_TB2")
        TAIL2_RB7 = Resnet_imagenet.get_layer(name="conv3_block4_out").output
        TAIL2_TB2_shortcut = Add()([TAIL2_TB2, TAIL2_RB7])

        TAIL2_TB3 = Transblock(input=TAIL2_TB2_shortcut, knum_out=256, layer_name="TAIL2_TB3")
        TAIL2_RB3 = Resnet_imagenet.get_layer(name="conv2_block3_out").output
        TAIL2_TB3_shortcut = Add()([TAIL2_TB3, TAIL2_RB3])

        TAIL2_TB4 = Transblock(input=TAIL2_TB3_shortcut, knum_out=128, layer_name="TAIL2_TB4")
        TAIL2_Trans_conv = Conv2DTranspose(filters=128, kernel_size=4, kernel_initializer="he_normal",
                                     activation="relu", strides=2, padding="same", name="TAIL2_Trans_conv")(TAIL2_TB4)
        TAIL2 = Conv2D(filters=64, kernel_size=3, kernel_initializer="he_normal",
                      activation="relu", strides=1, padding="same", name="TAIL2_conv")(TAIL2_Trans_conv)
        TAIL2 = Conv2D(filters=self.num_classes2, kernel_size=3, strides=1, padding="same", name="TAIL2_conv2")(TAIL2)
        output2 = Conv2D(filters=self.num_classes2, kernel_size=3, activation="softmax", strides=1, padding="same", name="output2")(TAIL2)
	
	    # option for comparing mIoU of this model with the main category model.
        to_main_cat = to_main_category(tensor=output2, to_category="mid2main")
        output3 = Activation(activation="softmax", name="ouput3")(to_main_cat)

        model = keras.Model(inputs=input, outputs=output3)
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            #loss={"output2" : SparseCategoricalCrossentropy()},
            loss = SparseCategoricalCrossentropy(),
            #metrics={"output2" : UpdatedMeanIoU(num_classes=self.num_classes2)}
            metrics=UpdatedMeanIoU(num_classes=8)
        )

        return model

class DlinkNet101_detail :

    def __init__(self, input_shape, num_classes3, imagenet_trainable, lr):
        self.input_shape = input_shape
        # Detail
        self.num_classes3 = num_classes3
        self.imagenet_trainable = imagenet_trainable
        self.lr = lr
        self.build_net()

    def build_net(self):

        input = keras.Input(self.input_shape)
        Resnet_imagenet = ResNet101(input_tensor=input, include_top=False, weights="imagenet")
        Resnet_imagenet.trainable = self.imagenet_trainable
        Resnet_output = Resnet_imagenet.output

        D_block = Dilated_layers(input=Resnet_output, knum=2048, ksize=3, layer_name="D_block")

        # TAIL3
        TAIL3_TB1 = Transblock(input=D_block, knum_out=1024, layer_name="TAIL3_TB1")
        TAIL3_RB30 = Resnet_imagenet.get_layer(name="conv4_block23_out").output
        TAIL3_TB1_shortcut = Add()([TAIL3_TB1, TAIL3_RB30])

        TAIL3_TB2 = Transblock(input=TAIL3_TB1_shortcut, knum_out=512, layer_name="TAIL3_TB2")
        TAIL3_RB7 = Resnet_imagenet.get_layer(name="conv3_block4_out").output
        TAIL3_TB2_shortcut = Add()([TAIL3_TB2, TAIL3_RB7])

        TAIL3_TB3 = Transblock(input=TAIL3_TB2_shortcut, knum_out=256, layer_name="TAIL3_TB3")
        TAIL3_RB3 = Resnet_imagenet.get_layer(name="conv2_block3_out").output
        TAIL3_TB3_shortcut = Add()([TAIL3_TB3, TAIL3_RB3])

        TAIL3_TB4 = Transblock(input=TAIL3_TB3_shortcut, knum_out=128, layer_name="TAIL3_TB4")
        TAIL3_Trans_conv = Conv2DTranspose(filters=128, kernel_size=4, kernel_initializer="he_normal",
                                     activation="relu", strides=2, padding="same", name="TAIL1_Trans_conv")(TAIL3_TB4)
        TAIL3 = Conv2D(filters=64, kernel_size=3, kernel_initializer="he_normal",
                      activation="relu", strides=1, padding="same", name="TAIL1_conv")(TAIL3_Trans_conv)
        output1 = Conv2D(filters=self.num_classes3, kernel_size=3,
                        activation="softmax", strides=1, padding="same", name="output1")(TAIL3)

        to_main_cat = to_main_category(tensor=output1, to_category="det2main")
        output2 = Activation(activation="softmax", name="output2")(to_main_cat)

        model = keras.Model(inputs=input, outputs=output2)
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss={"output2" : UpdatedSparseCategoricalCrossentropy()},
            metrics={"output2" : UpdatedMeanIoU(num_classes=self.num_classes3)}
        )

        return model

class DlinkNet101_3tails_v2 :

    def __init__(self, input_shape, num_classes1, num_classes2, num_classes3, resnet_trainable, lr):
        self.input_shape = input_shape
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.num_classes3 = num_classes3
        self.resnet_trainable = resnet_trainable
        self.lr = lr

    def build_net(self):

        input = keras.Input(self.input_shape)
        Resnet_imagenet = ResNet101(input_tensor=input, include_top=False, weights="imagenet")
        Resnet_imagenet.trainable = self.resnet_trainable
        Resnet_output = Resnet_imagenet.output

        D_block = Dilated_layers(input=Resnet_output, knum=2048, ksize=3, layer_name="D_block")

        tail_tb1 = Transblock(input=D_block, knum_out=1024, layer_name="TAIL_TB1")
        tail_rb1 = Resnet_imagenet.get_layer(name="conv4_block23_out").output
        tail_sc1 = Concatenate(axis=-1)([tail_tb1, tail_rb1])

        tail_tb2 = Transblock(input=tail_sc1, knum_out=512, layer_name="TAIL_TB2")
        tail_rb2 = Resnet_imagenet.get_layer(name="conv3_block4_out").output
        tail_sc2 = Concatenate(axis=-1)([tail_tb2, tail_rb2])

        tail_tb3 = Transblock(input=tail_sc2, knum_out=256, layer_name="TAIL_TB3")
        tail_rb3 = Resnet_imagenet.get_layer(name="conv2_block3_out").output
        tail_sc3 = Concatenate(axis=-1)([tail_tb3, tail_rb3])

        tail_tb4 = Transblock(input=tail_sc3, knum_out=128, layer_name="TAIL_TB4")

        # main
        tail_tp_main = Conv2DTranspose(filters=64, kernel_size=4, kernel_initializer="he_normal",
                                       activation="relu", strides=2, padding="same", name="TAIL_TP_MAIN")(tail_tb4)
        tail_main_output = Conv2D(filters=self.num_classes1, kernel_size=3,
                                  activation="softmax", strides=1, padding="same", name="TAIL_MAIN_OUTPUT")(tail_tp_main)
        # middle
        tail_tp_middle = Conv2DTranspose(filters=64, kernel_size=4, kernel_initializer="he_normal",
                                         activation="relu", strides=2, padding="same", name="TAIL_TP_MIDDLE")(tail_tb4)
        tail_middle_output = Conv2D(filters=self.num_classes2, kernel_size=3,
                                   activation="softmax", strides=1, padding="same", name="TAIL_MIDDLE_OUTPUT")(tail_tp_middle)
        # detail
        tail_tp_detail = Conv2DTranspose(filters=64, kernel_size=4, kernel_initializer="he_normal",
                                         activation="relu", strides=2, padding="same", name="TAIL_TP_DETAIL")(tail_tb4)
        tail_detail_output = Conv2D(filters=self.num_classes3, kernel_size=3,
                                   activation="softmax", strides=1, padding="same", name="TAIL_DETAIL_OUTPUT")(tail_tp_detail)

        model = keras.Model(inputs=input, outputs=[tail_main_output, tail_middle_output, tail_detail_output])
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss={"TAIL_MAIN_OUTPUT" : SparseCategoricalCrossentropy(), "TAIL_MIDDLE_OUTPUT" : SparseCategoricalCrossentropy(), "TAIL_DETAIL_OUTPUT" : SparseCategoricalCrossentropy()},
            metrics={"TAIL_MAIN_OUTPUT" : UpdatedMeanIoU(num_classes=self.num_classes1), "TAIL_MIDDLE_OUTPUT" : UpdatedMeanIoU(num_classes=self.num_classes2), "TAIL_DETAIL_OUTPUT" : UpdatedMeanIoU(num_classes=self.num_classes3)}
        )
        return model


if __name__ == "__main__" :
    """
    input_shape = keras.Input((1024,1024,3))
    model = ResNet101(input_tensor=input_shape, include_top=False, weights="imagenet")
    model.summary()
    """
    seed=10
    BATCH_SIZE=1
    image_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90
    )
    # mask_datagen_main = ImageDataGenerator(
    #     horizontal_flip=True,
    #     vertical_flip=True,
    #     rotation_range = 90
    # )
    # mask_datagen_mid = ImageDataGenerator(
    #     horizontal_flip=True,
    #     vertical_flip=True,
    #     rotation_range = 90
    # )
    mask_datagen_det = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90
    )

    # valid generator
    # validimg_datagen = ImageDataGenerator()
    # validmsk_datagen = ImageDataGenerator()

    # test generator
    # test_datagen = ImageDataGenerator(
    #     rescale=1./255
    # )

    #train
    image_generator = image_datagen.flow_from_directory(
        directory="E:\\Data_list\\Deep_learning_dataset\\Landcover\\train\\image",
        class_mode=None,
        seed=seed,
        shuffle=True,
        target_size=(1024, 1024),
        batch_size=BATCH_SIZE
    )

    # mask_generator_main = mask_datagen_main.flow_from_directory(
    #    directory="Dataset/train/mask_main",
    #    class_mode=None,
    #    seed=seed,
    #    shuffle=True,
    #    target_size=(1024, 1024),
    #    color_mode="grayscale",
    #     batch_size=BATCH_SIZE
    # )
    # mask_generator_middle = mask_datagen_mid.flow_from_directory(
    #    directory="Dataset/train/mask_middle",
    #    class_mode=None,
    #    seed=seed,
    #    shuffle=True,
    #    target_size=(1024, 1024),
    #    color_mode="grayscale",
    #    batch_size=BATCH_SIZE
    # )

    mask_generator_det = mask_datagen_det.flow_from_directory(
        directory="E:\\Data_list\\Deep_learning_dataset\\Landcover\\train\\mask_detail",
        class_mode=None,
        seed=seed,
        shuffle=True,
        target_size=(1024, 1024),
        color_mode="grayscale",
        batch_size=BATCH_SIZE
    )

    # test_generator = test_datagen.flow_from_directory(
    #     directory="E:\\Data_list\\Deep_learning_dataset\\Landcover\\valid\\image",
    #     target_size=(1024, 1024),
    #    shuffle=False,
    #     batch_size=1
    # )

    # # 3 tail generator
    # def generator(image_generator, mask_generator_main, mask_generator_mid, mask_generator_det) :
    #     while True :
    #         for x, y1, y2, y3 in zip(image_generator, mask_generator_main, mask_generator_mid, mask_generator_det) :
    #             yield x, (y1, y2, y3)
    #
    # train_gen = generator(image_generator, mask_generator_main, mask_generator_mid, mask_generator_det)

    # # valid
    # pred_img_generator=validimg_datagen.flow_from_directory(
    #     directory="E:\\Data_list\\Deep_learning_dataset\\Landcover\\valid\\image",
    #     class_mode=None,
    #     seed=seed,
    #     shuffle=True,
    #     target_size=(1024, 1024),
    #     batch_size=BATCH_SIZE
    # )

    # pred_msk_generator=validmsk_datagen.flow_from_directory(
    #     directory="E:\\Data_list\\Deep_learning_dataset\\Landcover\\valid\\mask",
    #     class_mode=None,
    #     seed=seed,
    #     shuffle=True,
    #     target_size=(1024, 1024),
    #     color_mode="grayscale",
    #     batch_size=BATCH_SIZE
    # )
    #
    train_gen = zip(image_generator, mask_generator_det)
    # valid_gen = zip(pred_img_generator, pred_msk_generator)

    # callbacks = [
    #     keras.callbacks.ModelCheckpoint(filepath=".\\DlinkNet_weights\\DlinkNet34\\DlinkNet34_weight2_10^-3{epoch:02d}.hdf5"),
    #     keras.callbacks.TensorBoard(log_dir=".\\logs\\DlinkNet34", update_freq="batch"),
	#     keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    # ]

    # # 3 tail
    # DlinkNet = DlinkNet101_3tails(input_shape=(1024, 1024, 3), num_classes1=8, num_classes2=23, num_classes3=42, imagenet_trainable=True, lr=0.0015)
    
    # middle
    # DlinkNet = DlinkNet101_middle(input_shape=(1024, 1024, 3), num_classes2=23, imagenet_trainable=True, lr=0.0005)
    
    # detail
    # DlinkNet = DlinkNet101_detail(input_shape=(1024, 1024, 3), num_classes2=23, imagenet_trainable=True, lr=0.0005)

    # test
    # DlinkNet = DlinkNet101_detail(input_shape=(1024, 1024, 3), num_classes3=42, imagenet_trainable=True, lr=0.0001)
    DlinkNet = DlinkNet34(input_shape=(1024, 1024, 3), num_classes=42)
    model = DlinkNet.build_net()
    #model.summary()
    # model.load_weights("E:\\Data_list\\DeepLearning_Model\\Semantic_segmentation_git\\DlinkNet_weights\\DlinkNet101_detail_weights\\DlinkNet_detail_40_76.hdf5")
    # class_weight_dict = {0:1.29, 1:2.00, 2:1.96, 3:1.10, 4:213.81,
    #                      5:8.32, 6:121538.63, 7:31.57, 8:13.31, 9:0.21,
    #                      10:10, 11:105.87, 12:7.64, 13:6.27, 14:0.25,
    #                      15:0.47, 16:5.45, 17:0.29, 18:2.90, 19:4.12,
    #                      20:9.89, 21:9.62, 22:0.15, 23:0.22, 24:0.60,
    #                      25:8.41, 26:19.75, 27:2.01, 28:0.25, 29:0.92,
    #                      30:15.57, 31:14.90, 32:401.19, 33:26.37, 34:5.40,
    #                      35:61.79, 36:8.32, 37:0.68, 38:2.35, 39:2.30,
    #                      40:1.35, 41:1.00}
    # sample_weight_np = np.ndarray([1.29, 2.00, 1.96, 1.10, 213.81,
    #                                8.32, 121538.63, 31.57, 13.31, 0.21,
    #                                10, 105.87, 7.64, 6.27, 0.25,
    #                                0.47, 5.45, 0.29, 2.90, 4.12,
    #                                9.89, 9.62, 0.15, 0.22, 0.60,
    #                                8.41, 19.75, 2.01, 0.25, 0.92,
    #                                15.57, 14.90, 401.19, 26.37, 5.40,
    #                                61.79, 8.32, 0.68, 2.35, 2.30,
    #                                1.35, 1.00])
    # model fitting

    model.fit(
        train_gen,
        steps_per_epoch=mask_generator_det.samples/BATCH_SIZE,
        epochs=20
    #    callbacks=[callbacks],
    )

    # # model evaluate
    # model.evaluate(
    #     valid_gen,
    #     batch_size=BATCH_SIZE
    # )

    # model predict
    for data, name in zip(test_generator, test_generator.filenames) :
        name = str(name).split('/')[-1].split('\\')[-1].split('.')[0]
        landcover_test = model.predict(
           data,
           batch_size=1,
           verbose=1,
           steps=test_generator.samples/1.
        )
        cv2.imwrite('.\\test_result\\test.png', landcover_test)

        class2color(landcover_test, name)