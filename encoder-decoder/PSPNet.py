import sys
sys.path.append("./backbones")
from utils import *
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Concatenate, Activation
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet101
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

class PSPNet :

    def __init__(self, input_shape, num_classes, backbone_trainable):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.backbone_trainable = backbone_trainable

    def build_net(self):

        # BackBone : ResNet101 pretrained with ImageNet
        input = keras.Input(self.input_shape)
        backbone = ResNet101(input_tensor=input, include_top=False, weights="imagenet")
        backbone.trainable=self.backbone_trainable
        backbone_feature = backbone.output

        PSP_module_output = Pyramid_module(input=backbone_feature)
        print(PSP_module_output.shape)
        conv = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same")(PSP_module_output)
        conv_batch = BatchNormalization()(conv)
        conv_activ = Activation(activation="relu")(conv_batch)
        print(conv_activ.shape)
        conv2 = Conv2D(filters=self.num_classes, kernel_size=(1,1), strides=(1,1))(conv_activ)
        conv2_upsample = UpSampling2D(size=(32,32), interpolation="bilinear")(conv2)
        print(conv2_upsample.shape)
        model = keras.Model(inputs=input, outputs=conv2_upsample)
        model.compile(
            optimizer = keras.optimizers.Adam(0.0001),
            loss = SparseCategoricalCrossentropy(),
            metrics = MeanIoU(num_classes=self.num_classes)
        )

        return model


if __name__ == "__main__" :

    INPUT_SHAPE = (512, 512, 3)
    NUM_CLASSES = 5
    TRAINABLE = True

    PSP = PSPNet(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES, backbone_trainable=TRAINABLE)
    model = PSP.build_net()
    model.summary()