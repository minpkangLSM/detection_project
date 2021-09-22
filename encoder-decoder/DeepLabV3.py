import sys
sys.path.append("./backbones")
from utils import *
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Activation
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.keras.applications import Xception
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
    class_weight = [1/0.175, 1/0.287, 1/0.123, 1/0.257, 1/0.137, 1/0.009, 1/0.011, 1/0.001]
    weight = class_weight_matrix(y_true, class_weight)
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

class DLV3_ResNet :
    """
    # Check points in DeepLabV3

    ## Encoder part
    1. ResNet_DLV3의 Resblock mod의 필터 수는 어떻게 설정해야 하는가?
    2. ResNet_DLV3_atrous의 dilation_rate는 어떻게 설정해야 하는가?(DLV3의 unit_rate 설정 방법)

    ## Decoder part
    3. Bilinear Upsampling의 정확한 방법은? (지금 Deconvolution으로 구현)

    4. 학습 때와 테스트 때의 output_stride의 크기를 변경하기 위해서는 어떻게 구현해야 하는가?
    """

    def __init__(self, input_shape, num_classes, lr, unit_rate):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lr = lr
        self.u_rate = unit_rate

    def build_net_cascade(self):

        input = keras.Input(shape=self.input_shape)

        # DCNN - Encoder
        conv1 = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding="same", name="conv1")(input)
        pool1 = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same", name="pool1")(conv1)

        Rb1 = Resblock_DLV3(input=pool1, f1=64, f2=64, f3=64, verbose="False", block_name="Rb1")
        Rb2 = Resblock_DLV3(input=Rb1, f1=128, f2=128, f3=128, verbose="False", block_name="Rb2")
        Rb3 = Resblock_DLV3(input=Rb2, f1=256, f2=256, f3=256, verbose="True", block_name="Rb3")

        Rb4 = Resblock_DLV3_atrous(input=Rb3, f1=256, f2=256, f3=256,
                                   verbose="False", u_rate=self.u_rate, c_rate=2, block_name="Rb4_atrous")
        Rb5 = Resblock_DLV3_atrous(input=Rb4, f1=256, f2=256, f3=256,
                                   verbose="False", u_rate=self.u_rate, c_rate=4, block_name="Rb5_atrous")
        Rb6 = Resblock_DLV3_atrous(input=Rb5, f1=256, f2=256, f3=256,
                                   verbose="False", u_rate=self.u_rate, c_rate=8, block_name="Rb6_atrous")
        Rb7 = Resblock_DLV3_atrous(input=Rb6, f1=256, f2=256, f3=256,
                                   verbose="True", u_rate=self.u_rate, c_rate=16, block_name="R7_atrous")

        # Decoder
        # GAP(Global Average Pooling)?
        last_conv = Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding="same", name="last_conv")(Rb7)
        last_batch = BatchNormalization()(last_conv)
        # need to convert into Bilinear upsampling
        dconv = Conv2DTranspose(filters=self.num_classes, kernel_size=(3,3), strides=(16,16), activation="softmax")(last_batch)


        model = keras.Model(inputs=input, outputs=dconv)
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=SparseCategoricalCrossentropy(),
            metrics=UpdatedMeanIoU(num_classes=self.num_classes)
        )

        return model

    def build_net_parallel(self):

        input = keras.Input(shape=self.input_shape)

        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same", name="conv1")(input)
        pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="pool1")(conv1)

        Rb1 = Resblock_DLV3(input=pool1, f1=64, f2=64, f3=64, verbose="False", block_name="Rb1")
        Rb2 = Resblock_DLV3(input=Rb1, f1=128, f2=128, f3=128, verbose="False", block_name="Rb2")
        Rb3 = Resblock_DLV3(input=Rb2, f1=256, f2=256, f3=256, verbose="True", block_name="Rb3")

        Rb4 = Resblock_DLV3_atrous(input=Rb3, f1=256, f2=256, f3=256,
                                   verbose="False", u_rate=self.u_rate, c_rate=2, block_name="Rb4_atrous")

        # ASPP(Atrous Spatial Pyramid Pooling)
        ASPP_conv1 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same", name="ASPP_conv1")(Rb4)
        ASPP_conv1_batch = BatchNormalization()(ASPP_conv1)
        ASPP_conv2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=6, padding="same",
                            name="ASPP_conv2")(Rb4)
        ASPP_conv2_batch = BatchNormalization()(ASPP_conv2)
        ASPP_conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=12, padding="same",
                            name="ASPP_conv3")(Rb4)
        ASPP_conv3_batch = BatchNormalization()(ASPP_conv3)
        ASPP_conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=18, padding="same",
                            name="ASPP_conv4")(Rb4)
        ASPP_conv4_batch = BatchNormalization()(ASPP_conv4)
        # Concat : Image-level feature == Rb4??
        Concat = Concatenate()([ASPP_conv1_batch, ASPP_conv2_batch, ASPP_conv3_batch, ASPP_conv4_batch, Rb4])

        # Decoder
        # GAP(Global Average Pooling)?
        last_conv1 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same", name="last_conv1")(Concat)
        last_batch = BatchNormalization()(last_conv1)
        # need to convert into Bilinear upsampling
        dconv = Conv2DTranspose(filters=self.num_classes, kernel_size=(3, 3), strides=(16, 16), activation="softmax")(
            last_batch)

        model = keras.Model(inputs=input, outputs=dconv)
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=SparseCategoricalCrossentropy(),
            metrics=UpdatedMeanIoU(num_classes=self.num_classes)
        )

        return model

class DLV3P_Xception :
    """
    1. Xception의 각 module에 들어가는 unit_rate, corresponding_rate는 어떻게 되는가?
       - output stride를 통해 추론할 수 있나?
       - 다른 구현 사례 통해 확인 가능
    2. Global average pooling은 어디에 위치해야 하는가? - 구현 및 체크 완료
    3. Image-level features는 무엇인가? - 구현 및 체크 완료
    4. Upsampling에 있어 Conv2DTranspose? Upsampling(interpolation="bilinear")? - bilinear interpolation
    """

    def __init__(self, input_shape, num_classes, lr, unit_rate):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lr = lr
        self.u_rate = unit_rate

    def build_net(self):

        # Encoder part
        dilation_rate_temp = 2

        input = keras.Input(shape=self.input_shape)

        # Entry flow
        conv1 = Conv2D(filters=32, kernel_size=(3, 3),
                       strides=(2, 2), padding="same", name="conv1")(input)
        conv1_batch = BatchNormalization()(conv1)
        conv1_active = Avtivation(activation="relu")(conv1_batch)

        conv2 = _conv2d_same(conv1_active, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
        conv2_batch = BatchNormalization()(conv2)
        conv2_active = Activation(activation="relu")(conv2_batch)

        En1 = X_module_entry_DLV3P(input=conv2_active, knum=128, u_rate=self.u_rate, c_rate=1, weight_decay=0.0002, block_name="En1")
        En2 = X_module_entry_DLV3P(input=En1, knum=256, u_rate=self.u_rate, c_rate=1, weight_decay=0.0002,  block_name="En2")
        En3 = X_module_entry_DLV3P(input=En2, knum=728, u_rate=self.u_rate, c_rate=1, weight_decay=0.0002, block_name="En3")

        # Middle flow
        Mi1 = X_module_middle_DLV3P(input=En3, knum=728, u_rate=self.u_rate, c_rate=1, weight_decay=0.0002, block_name="Mi1")
        Mi2 = X_module_middle_DLV3P(input=Mi1, knum=728, u_rate=self.u_rate, c_rate=1, weight_decay=0.0002, block_name="Mi2")
        Mi3 = X_module_middle_DLV3P(input=Mi2, knum=728, u_rate=self.u_rate, c_rate=1, weight_decay=0.0002, block_name="Mi3")
        Mi4 = X_module_middle_DLV3P(input=Mi3, knum=728, u_rate=self.u_rate, c_rate=1, weight_decay=0.0002, block_name="Mi4")
        Mi5 = X_module_middle_DLV3P(input=Mi4, knum=728, u_rate=self.u_rate, c_rate=1, weight_decay=0.0002, block_name="Mi5")
        Mi6 = X_module_middle_DLV3P(input=Mi5, knum=728, u_rate=self.u_rate, c_rate=1, weight_decay=0.0002, block_name="Mi6")
        Mi7 = X_module_middle_DLV3P(input=Mi6, knum=728, u_rate=self.u_rate, c_rate=1, weight_decay=0.0002, block_name="Mi7")
        Mi8 = X_module_middle_DLV3P(input=Mi7, knum=728, u_rate=self.u_rate, c_rate=1, weight_decay=0.0002, block_name="Mi8")
        Mi9 = X_module_middle_DLV3P(input=Mi8, knum=728, u_rate=self.u_rate, c_rate=1, weight_decay=0.0002, block_name="Mi9")
        Mi10 = X_module_middle_DLV3P(input=Mi9, knum=728, u_rate=self.u_rate, c_rate=1, weight_decay=0.0002, block_name="Mi10")
        Mi11 = X_module_middle_DLV3P(input=Mi10, knum=728, u_rate=self.u_rate, c_rate=1, weight_decay=0.0002, block_name="Mi11")
        Mi12 = X_module_middle_DLV3P(input=Mi11, knum=728, u_rate=self.u_rate, c_rate=1, weight_decay=0.0002, block_name="Mi12")
        Mi13 = X_module_middle_DLV3P(input=Mi12, knum=728, u_rate=self.u_rate, c_rate=1, weight_decay=0.0002, block_name="Mi13")
        Mi14 = X_module_middle_DLV3P(input=Mi13, knum=728, u_rate=self.u_rate, c_rate=1, weight_decay=0.0002, block_name="Mi14")
        Mi15 = X_module_middle_DLV3P(input=Mi14, knum=728, u_rate=self.u_rate, c_rate=1, weight_decay=0.0002, block_name="Mi15")
        Mi16 = X_module_middle_DLV3P(input=Mi15, knum=728, u_rate=self.u_rate, c_rate=1, weight_decay=0.0002, block_name="Mi16")

        # Exit flow
        Ex1 = X_module_exit_DLV3P(input=Mi16, knum_in=728, knum_out=1024, u_rate=self.u_rate, c_rate=1, weight_decay=0.0002, block_name="Ex1")
        Ex1_conv1 = SeparableConv2D(filters=1536, kernel_size=(3, 3), kernel_initializer="he_normal",
                                    strides=(1, 1), dilation_rate=dilation_rate_temp, padding="same", name="Ex1_conv1", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(
            Ex1)
        Ex1_batch1 = BatchNormalization()(Ex1_conv1)
        Ex1_active1 = Activation(activation="relu", name="Ex1_active1")(Ex1_batch1)

        Ex1_conv2 = SeparableConv2D(filters=1536, kernel_size=(3, 3), kernel_initializer="he_normal",
                                    strides=(1, 1), dilation_rate=dilation_rate_temp, padding="same", name="Ex1_conv2", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(
            Ex1_active1)
        Ex1_batch2 = BatchNormalization()(Ex1_conv2)
        Ex1_active2 = Activation(activation="relu", name="Ex1_active2")(Ex1_batch2)

        Ex1_conv3 = SeparableConv2D(filters=2048, kernel_size=(3, 3), kernel_initializer="he_normal",
                                    strides=(1, 1), dilation_rate=dilation_rate_temp, padding="same", name="Ex1_conv3", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(
            Ex1_active2)
        Ex1_batch3 = BatchNormalization()(Ex1_conv3)
        Ex1_active3 = Activation(activation="relu", name="Ex1_active3")(Ex1_batch3)

        # ASPP(Atrous Spatial Pyramid Pooling)
        ASPP_conv1 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same",
                            name="ASPP_conv1", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(Ex1_active3)
        ASPP_conv1_batch = BatchNormalization()(ASPP_conv1)
        ASPP_conv2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=6, padding="same",
                            name="ASPP_conv2", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(Ex1_active3)
        ASPP_conv2_batch = BatchNormalization()(ASPP_conv2)
        ASPP_conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=12, padding="same",
                            name="ASPP_conv3", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(Ex1_active3)
        ASPP_conv3_batch = BatchNormalization()(ASPP_conv3)
        ASPP_conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=18, padding="same",
                            name="ASPP_conv4", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(Ex1_active3)
        ASPP_conv4_batch = BatchNormalization()(ASPP_conv4)

        # image level feature
        ASPP5 = GlobalAvgPool2D()(Ex1_active3)
        ASPP5 = K.expand_dims(ASPP5, axis=-1)
        ASPP5 = K.expand_dims(ASPP5, axis=-1)
        ASPP5 = tf.transpose(ASPP5, [0, 3, 2, 1])
        ASPP5 = UpSampling2D(size=tuple(Ex1_active1.shape[1:3]), interpolation="bilinear")(ASPP5)

        Concat = Concatenate()(
            [ASPP_conv1_batch, ASPP_conv2_batch, ASPP_conv3_batch, ASPP_conv4_batch, Ex1_active3, ASPP5])

        conv1 = Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding="same", name="1x1_after_concat", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(Concat)
        # Conv2DTranspose vs UpSampling2D
        conv1_deconv = UpSampling2D(size=(4,4), interpolation="bilinear")(conv1)
        # conv1_deconv = Conv2DTranspose(filters=256, kernel_size=(3,3), strides=(4,4), padding="same")(conv1)

        # Decoder part
        # "En2_act3" : output_stride=8, channel_depth=728
        low_level_feature = En2
        low_level_feature_conv1 = Conv2D(filters=48, kernel_size=(1,1), strides=(1,1), padding="same", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(low_level_feature)

        Concat2 = Concatenate()([conv1_deconv, low_level_feature_conv1])
        Concat2_conv3 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(Concat2)
        Concat2_conv3 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(Concat2_conv3)
        Concat2_conv3_2 = Conv2D(filters=self.num_classes, kernel_size=(3,3), strides=(1,1), padding="same", kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(Concat2_conv3)

        # Conv2DTranspose vs UpSampling2D
        Concat2_deconv = UpSampling2D(size=(8,8), interpolation="bilinear")(Concat2_conv3_2)
        #Concat2_deconv = Conv2DTranspose(filters=self.num_classes, kernel_size=(3,3), strides=(8,8))(Concat2_conv3_2)

        model = keras.Model(inputs=input, outputs=Concat2_deconv)
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=SparseCategoricalCrossentropy(),
            metrics=UpdatedMeanIoU(num_classes=self.num_classes)
        )

        return model


class DLV3P_Xception_pretrained :

    def __init__(self, input_shape, num_classes, lr, backbone_trainable):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lr = lr
        self.backbone_trainable=backbone_trainable

    def build_net(self):

        input=keras.Input(self.input_shape)
        backbone = Xception(input_tensor=input, include_top=False, weights="imagenet")
        # for layer in backbone.layers :
        #     print(layer)
        #     layer.kernel_regularizer = keras.regularizers.L1L2(0.0001)
        backbone.trainable=self.backbone_trainable
        backbone_feature = backbone.output

        # ASPP(Atrous Spatial Pyramid Pooling)
        ASPP_conv1 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same",
                            name="ASPP_conv1")(backbone_feature)
        ASPP_conv1_batch = BatchNormalization()(ASPP_conv1)
        ASPP_conv2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=6, padding="same",
                            name="ASPP_conv2")(backbone_feature)
        ASPP_conv2_batch = BatchNormalization()(ASPP_conv2)
        ASPP_conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=12, padding="same",
                            name="ASPP_conv3")(backbone_feature)
        ASPP_conv3_batch = BatchNormalization()(ASPP_conv3)
        ASPP_conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=18, padding="same",
                            name="ASPP_conv4")(backbone_feature)
        ASPP_conv4_batch = BatchNormalization()(ASPP_conv4)

        # image level feature
        ASPP5 = GlobalAvgPool2D()(backbone_feature)
        ASPP5 = K.expand_dims(ASPP5, axis=-1)
        ASPP5 = K.expand_dims(ASPP5, axis=-1)
        ASPP5 = tf.transpose(ASPP5, [0, 3, 2, 1])
        ASPP5 = UpSampling2D(size=tuple(backbone_feature.shape[1:3]), interpolation="bilinear")(ASPP5)

        Concat = Concatenate()(
            [ASPP_conv1_batch, ASPP_conv2_batch, ASPP_conv3_batch, ASPP_conv4_batch, backbone_feature, ASPP5])

        conv1 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same", name="1x1_after_concat",
                       kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(Concat)
        # Conv2DTranspose vs UpSampling2D
        conv1_deconv = UpSampling2D(size=(4, 4), interpolation="bilinear")(conv1)
        # conv1_deconv = Conv2DTranspose(filters=256, kernel_size=(3,3), strides=(4,4), padding="same")(conv1)

        block4_sepconv2 = backbone.get_layer(name="block4_sepconv2")

        block4_sepconv2 = backbone.get_layer(name="block4_sepconv2").output
        block4_sepconv2_shrk = Conv2D(filters=48, kernel_size=(1, 1), strides=(1, 1), padding="same",
                                      kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(block4_sepconv2)

        Concat2 = Concatenate()([conv1_deconv, block4_sepconv2_shrk])
        Concat2_conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same",
                               kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(Concat2)
        Concat2_conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same",
                               kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(Concat2_conv3)
        Concat2_conv3_2 = Conv2D(filters=self.num_classes, kernel_size=(3, 3), strides=(1, 1), padding="same",
                                 kernel_regularizer=tf.keras.regularizers.L1L2(0.00004))(Concat2_conv3)

        # Conv2DTranspose vs UpSampling2D
        Concat2_deconv = UpSampling2D(size=(8, 8), interpolation="bilinear")(Concat2_conv3_2)
        # Concat2_deconv = Conv2DTranspose(filters=self.num_classes, kernel_size=(3,3), strides=(8,8))(Concat2_conv3_2)

        model = keras.Model(inputs=input, outputs=Concat2_deconv)
        model.compile(
            optimizer = keras.optimizers.Adam(self.lr),
            loss = SparseCategoricalCrossentropy(),
            metrics = [UpdatedMeanIoU(num_classes=self.num_classes)]
        )

        return model

if __name__ == "__main__" :
    seed = 10
    BATCH_SIZE = 1
    INPUT_SHAPE = (512, 512, 3)
    NUM_CLASSES = 33
    LEARNING_RATE = 0.00001
    UNIT_RATE = (1,1,1)

    image_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90
    )
    mask_datagen_det = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90
    )

    # train
    image_generator = image_datagen.flow_from_directory(
        directory="E:\\Data_list\\Deep_learning_dataset\\Cityscapes\\dataset\\dataset_mod\\train\\images",
        class_mode=None,
        seed=seed,
        shuffle=True,
        target_size=(512, 512),
        batch_size=BATCH_SIZE
    )
    mask_generator_det = mask_datagen_det.flow_from_directory(
        directory="E:\\Data_list\\Deep_learning_dataset\\Cityscapes\\dataset\\dataset_mod\\train\\labels",
        class_mode=None,
        seed=seed,
        shuffle=True,
        target_size=(512, 512),
        color_mode="grayscale",
        batch_size=BATCH_SIZE
    )

    train_gen = zip(image_generator, mask_generator_det)

    # callbacks = [
    #     keras.callbacks.ModelCheckpoint(filepath=".\\DlinkNet_weights\\DlinkNet34\\DlinkNet34_weight2_10^-3{epoch:02d}.hdf5"),
    #     keras.callbacks.TensorBoard(log_dir=".\\logs\\DlinkNet34", update_freq="batch"),
    #     keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    # ]

    #DLV3 = DLV3P_Xception(INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE, UNIT_RATE)
    DLV3_P = DLV3P_Xception_pretrained(INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE, True)
    model = DLV3_P.build_net()
    model.summary()

    model.fit(
        train_gen,
        steps_per_epoch=mask_generator_det.samples/BATCH_SIZE,
        batch_size=BATCH_SIZE,
        epochs=20
    )