import os
import numpy as np
from tifffile import tifffile as tifi
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import multiply
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import MeanIoU
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

# Optionset
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
session = tf.compat.v1.InteractiveSession(config=config)

class UpdatedMeanIoU(MeanIoU) :
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

def class2color(mask, name):
    mask = mask.reshape(512, 512, 2)
    # mask = np.argmax(mask, axis=-1)
    # mask_zero = np.zeros([512, 512, 3])
    # name = name+".png"
    # for class_num in range(2):
    #     mask_label = mask == class_num
    #     if class_num == 0:
    #         mask_zero[mask_label] = [0, 0, 0]
    #     elif class_num == 1:
    #         mask_zero[mask_label] = [255, 255, 255]
    #
    # print(name)
    # im = Image.fromarray(mask_zero.astype(np.uint8))
    # im.save("./data_for_sangam_pred/dist2_pred/"+name, 'png')
    tifi.imwrite("./data_for_sangam_pred/valid/DRU/"+ name+".tif", mask[:, :, 1])

def Resblock_bn_DRU(input_tensor, channels, weight_decay = None):

    conv1 = Conv2D(filters=channels,
                   kernel_size=(3,3),
                   kernel_initializer="he_normal",
                   padding="same",
                   trainable=False
                   )(input_tensor)
    batchnorm1 = BatchNormalization(trainable=False)(conv1)
    active1 = Activation(activation="relu")(batchnorm1)

    conv2 = Conv2D(filters = 2*channels,
                   kernel_size=(3,3),
                   kernel_initializer="he_normal",
                   padding="same",
                   trainable=False
                   )(active1)
    batchnorm2 = BatchNormalization(trainable=False)(conv2)
    active2 = Activation(activation="relu")(batchnorm2)

    conv3 = Conv2D(filters=2*channels,
                   kernel_size=(1,1),
                   kernel_initializer="he_normal",
                   padding="same",
                   trainable=False
                   )(active2)
    batchnorm3 = BatchNormalization(trainable=False)(conv3)
    residual = Add()([input_tensor, batchnorm3])
    active3 = Activation(activation="relu")(residual)

    return active3

class softmax_temperature(layers.Layer):
    def __init__(self, batch_size, axis=-1):
        super(softmax_temperature, self).__init__()
        self.units=batch_size
        self.axis = axis

    def build(self, input_shape):
        self.scale = self.add_weight(initializer=tf.initializers.ones(),
                                     trainable=True)
    def get_config(self):
        config = super().get_config().copy()
        return config

    def call(self, x):
        rank = x.shape.rank
        x = x / self.scale
        if rank == 2:
            output = nn.softmax(x)
        elif rank > 2:
            e = math_ops.exp(x - math_ops.reduce_max(x, axis=self.axis, keepdims=True))
            s = math_ops.reduce_sum(e, axis=self.axis, keepdims=True)
            output = e / s
        else:
            raise ValueError('Cannot apply softmax to a tensor that is 1D. '
                             'Received input: %s' % (x,))
        # Cache the logits to use for crossentropy loss.
        output._keras_logits = x  # pylint: disable=protected-access
        return output

class DeepResUNet :
    """
    CODE INFO
    """

    def __init__(self, input_size, lr, num_classes, batch_size):
        self.input_size = input_size
        self.lr = lr
        self.num_classes = num_classes
        self.batch_size = batch_size

    def pretrained(self):

        input = tf.keras.Input(self.input_size)

        # Encoder - Input parts : 5x5 - Pool2
        En_conv5x5 = Conv2D(filters=128,
                            kernel_size=(5, 5),
                            kernel_initializer="he_normal",
                            padding="same",
                            trainable=False
                            )(input)
        En_Conv5x5_bn = BatchNormalization(trainable=False)(En_conv5x5)
        En_Max2x2_1 = MaxPool2D(pool_size=(2, 2),
                                strides=(2, 2)
                                )(En_Conv5x5_bn)
        # block1 : Resblock x 2, Pool2
        En_rb1 = Resblock_bn_DRU(En_Max2x2_1,
                                 64)
        En_rb2 = Resblock_bn_DRU(En_rb1,
                                 64)
        En_add1 = Add()([En_Max2x2_1, En_rb2])
        En_pool1 = MaxPool2D(pool_size=(2, 2),
                             strides=(2, 2)
                             )(En_add1)
        # block2 : Resblock x 2, Pool2
        En_rb3 = Resblock_bn_DRU(En_pool1,
                                 64)
        En_rb4 = Resblock_bn_DRU(En_rb3,
                                 64)
        En_add2 = Add()([En_rb4, En_pool1])
        En_pool2 = MaxPool2D(pool_size=(2, 2),
                             strides=(2, 2)
                             )(En_add2)
        # block3 : Resblock x 2, Pool2
        En_rb5 = Resblock_bn_DRU(En_pool2,
                                 64)
        En_rb6 = Resblock_bn_DRU(En_rb5,
                                 64)
        En_add3 = Add()([En_rb6, En_pool2])
        En_pool3 = MaxPool2D(pool_size=(2, 2),
                             strides=(2, 2)
                             )(En_add3)
        # block4 : Resblock x 2
        En_rb7 = Resblock_bn_DRU(En_pool3,
                                 64)
        En_rb8 = Resblock_bn_DRU(En_rb7,
                                 64)
        En_add4 = Add()([En_rb8, En_pool3])

        De_up1 = Conv2DTranspose(
            filters=128,
            kernel_size=(2, 2),
            strides=(2, 2),
            trainable=False
        )(En_add4)
        De_concat1 = Concatenate()([De_up1, En_add3])
        De_conv1x1_1 = Conv2D(
            filters=128,
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer="he_normal",
            trainable=False
        )(De_concat1)
        De_rb1 = Resblock_bn_DRU(
            De_conv1x1_1,
            64
        )
        De_rb2 = Resblock_bn_DRU(
            De_rb1,
            64
        )
        # block6
        De_up2 = Conv2DTranspose(
            filters=128,
            kernel_size=(2, 2),
            strides=(2, 2),
            trainable=False
        )(De_rb2)
        De_concat2 = Concatenate()([De_up2, En_add2])
        De_conv1x1_2 = Conv2D(
            filters=128,
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer="he_normal",
            trainable=False
        )(De_concat2)
        De_rb3 = Resblock_bn_DRU(
            De_conv1x1_2,
            64
        )
        De_rb4 = Resblock_bn_DRU(
            De_rb3,
            64
        )
        # block7
        De_up3 = Conv2DTranspose(
            filters=128,
            kernel_size=(2, 2),
            strides=(2, 2),
            trainable=False
        )(De_rb4)
        De_concat3 = Concatenate()([De_up3, En_add1])
        De_conv1x1_3 = Conv2D(
            filters=128,
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer="he_normal",
            trainable=False
        )(De_concat3)
        De_rb5 = Resblock_bn_DRU(
            De_conv1x1_3,
            64
        )
        De_rb6 = Resblock_bn_DRU(
            De_rb5,
            64
        )
        # block8
        De_up4 = Conv2DTranspose(
            filters=128,
            kernel_size=(2, 2),
            strides=(2, 2),
            trainable=False
        )(De_rb6)
        De_concat4 = Concatenate()([De_up4, En_conv5x5])
        De_conv1x1_4 = Conv2D(
            filters=128,
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer="he_normal",
            trainable=False
        )(De_concat4)
        De_rb7 = Resblock_bn_DRU(
            De_conv1x1_4,
            64
        )
        De_rb8 = Resblock_bn_DRU(
            De_rb7,
            64
        )
        De_conv1x1_5 = Conv2D(
            filters=self.num_classes,
            kernel_size=(1, 1),
            strides=(1, 1),
            trainable=False
        )(De_rb8)

        De_last = Activation(activation="softmax")(De_conv1x1_5)

        model = tf.keras.Model(inputs=input, outputs=De_last)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss=SparseCategoricalCrossentropy(),
            metrics=UpdatedMeanIoU(num_classes=self.num_classes)
        )

        return model

    def build_net(self):
        input = tf.keras.Input(self.input_size)

        # Encoder - Input parts : 5x5 - Pool2
        En_conv5x5 = Conv2D(filters=128,
                            kernel_size=(5, 5),
                            kernel_initializer="he_normal",
                            padding="same",
                            trainable=False
                            )(input)
        En_Conv5x5_bn = BatchNormalization(trainable=False)(En_conv5x5)
        En_Max2x2_1 = MaxPool2D(pool_size=(2, 2),
                                strides=(2, 2)
                                )(En_Conv5x5_bn)
        # block1 : Resblock x 2, Pool2
        En_rb1 = Resblock_bn_DRU(En_Max2x2_1,
                                 64)
        En_rb2 = Resblock_bn_DRU(En_rb1,
                                 64)
        En_add1 = Add()([En_Max2x2_1, En_rb2])
        En_pool1 = MaxPool2D(pool_size=(2, 2),
                             strides=(2, 2)
                             )(En_add1)
        # block2 : Resblock x 2, Pool2
        En_rb3 = Resblock_bn_DRU(En_pool1,
                                 64)
        En_rb4 = Resblock_bn_DRU(En_rb3,
                                 64)
        En_add2 = Add()([En_rb4, En_pool1])
        En_pool2 = MaxPool2D(pool_size=(2, 2),
                             strides=(2, 2)
                             )(En_add2)
        # block3 : Resblock x 2, Pool2
        En_rb5 = Resblock_bn_DRU(En_pool2,
                                 64)
        En_rb6 = Resblock_bn_DRU(En_rb5,
                                 64)
        En_add3 = Add()([En_rb6, En_pool2])
        En_pool3 = MaxPool2D(pool_size=(2, 2),
                             strides=(2, 2)
                             )(En_add3)
        # block4 : Resblock x 2
        En_rb7 = Resblock_bn_DRU(En_pool3,
                                 64)
        En_rb8 = Resblock_bn_DRU(En_rb7,
                                 64)
        En_add4 = Add()([En_rb8, En_pool3])

        De_up1 = Conv2DTranspose(
            filters=128,
            kernel_size=(2, 2),
            strides=(2, 2),
            trainable=False
        )(En_add4)
        De_concat1 = Concatenate()([De_up1, En_add3])
        De_conv1x1_1 = Conv2D(
            filters=128,
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer="he_normal",
            trainable=False
        )(De_concat1)
        De_rb1 = Resblock_bn_DRU(
            De_conv1x1_1,
            64
        )
        De_rb2 = Resblock_bn_DRU(
            De_rb1,
            64
        )
        # block6
        De_up2 = Conv2DTranspose(
            filters=128,
            kernel_size=(2, 2),
            strides=(2, 2),
            trainable=False
        )(De_rb2)
        De_concat2 = Concatenate()([De_up2, En_add2])
        De_conv1x1_2 = Conv2D(
            filters=128,
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer="he_normal",
            trainable=False
        )(De_concat2)
        De_rb3 = Resblock_bn_DRU(
            De_conv1x1_2,
            64
        )
        De_rb4 = Resblock_bn_DRU(
            De_rb3,
            64
        )
        # block7
        De_up3 = Conv2DTranspose(
            filters=128,
            kernel_size=(2, 2),
            strides=(2, 2),
            trainable=False
        )(De_rb4)
        De_concat3 = Concatenate()([De_up3, En_add1])
        De_conv1x1_3 = Conv2D(
            filters=128,
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer="he_normal",
            trainable=False
        )(De_concat3)
        De_rb5 = Resblock_bn_DRU(
            De_conv1x1_3,
            64
        )
        De_rb6 = Resblock_bn_DRU(
            De_rb5,
            64
        )
        # block8
        De_up4 = Conv2DTranspose(
            filters=128,
            kernel_size=(2, 2),
            strides=(2, 2),
            trainable=False
        )(De_rb6)
        De_concat4 = Concatenate()([De_up4, En_conv5x5])
        De_conv1x1_4 = Conv2D(
            filters=128,
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer="he_normal",
            trainable=False
        )(De_concat4)
        De_rb7 = Resblock_bn_DRU(
            De_conv1x1_4,
            64
        )
        De_rb8 = Resblock_bn_DRU(
            De_rb7,
            64
        )
        De_conv1x1_5 = Conv2D(
            filters=self.num_classes,
            kernel_size=(1, 1),
            strides=(1, 1),
            trainable=False
        )(De_rb8)

        De_last = softmax_temperature(batch_size=self.batch_size)(De_conv1x1_5)

        model = tf.keras.Model(inputs=input, outputs=De_last)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss=SparseCategoricalCrossentropy(),
            metrics=UpdatedMeanIoU(num_classes=self.num_classes)
        )

        return model

if __name__ == "__main__" :

    INPUT_SHAPE=(512, 512, 3)
    LR = 0.001
    NUM_CLASSES=2
    BATCH_SIZE = 4
    seed = 10

    # valid
    valid_img_datagen = ImageDataGenerator()
    valid_msk_datagen = ImageDataGenerator()

    valid_img_generator = valid_img_datagen.flow_from_directory(
        directory="./renewal_dataset_0324/valid/images",
        target_size=(512, 512),
        seed=seed,
        shuffle=True,
        class_mode=None,
        batch_size=BATCH_SIZE
    )
    valid_mask_generator = valid_msk_datagen.flow_from_directory(
        directory="./renewal_dataset_0324/valid/labels",
        target_size=(512, 512),
        seed=seed,
        shuffle=True,
        class_mode=None,
        color_mode="grayscale",
        batch_size=BATCH_SIZE
    )


    def create_train_generator(img, label):
        while True:
            for x1, x2 in zip(img, label):
                yield x1, x2

    # pred_img_datagen=ImageDataGenerator()
    #
    # pred_img_generator = pred_img_datagen.flow_from_directory(
    #     directory="./renewal_dataset_0324/valid/images",
    #     target_size=(512, 512),
    #     batch_size=1,
    #     shuffle=False,
    #     class_mode=None
    # )

    valid_gen = create_train_generator(valid_img_generator, valid_mask_generator)

    # pretrained model
    pretrained = DeepResUNet(input_size=INPUT_SHAPE, lr=LR, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE).pretrained()
    pretrained.load_weights("./renewal_dataset_0324_weight(Pretrained_finetune)/DRU_20.hdf5")

    # Calibrate model
    DeepResUNet = DeepResUNet(input_size=INPUT_SHAPE, lr=LR, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE)
    model = DeepResUNet.build_net()
    for i in range(len(pretrained.layers) - 1):
        model.layers[i].set_weights(pretrained.layers[i].get_weights())
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath="./DRU_calibrated_weights/DRU_calib_{epoch:02d}.hdf5")
    ]

    model.fit(
        valid_gen,
        steps_per_epoch = valid_mask_generator.samples/BATCH_SIZE,
        epochs=50,
        callbacks=callbacks
    )

    # for data, name in zip(pred_img_generator, pred_img_generator.filenames) :
    #     name = str(name).split('/')[-1].split('\\')[-1].split('.')[0]
    #     print(name)
    #     landcover_test = model.predict(
    #        data,
    #        batch_size=1,
    #        verbose=1,
    #        steps=pred_img_generator.samples/1.
    #     )
    #     class2color(landcover_test, name)