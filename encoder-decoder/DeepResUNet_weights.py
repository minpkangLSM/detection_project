import os
import numpy as np
import tensorflow as tf
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

def cust_loss(target, output):
    target = tf.cast(target, tf.uint8)
    target = tf.one_hot(target, 2)
    target = tf.squeeze(target, 3)
    return -tf.reduce_sum(target*output, len(output.get_shape())-1)

def dice_loss(y_true, y_pred):

    y_true = tf.cast(y_true, tf.int64)
    y_true = K.one_hot(y_true, 2)

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = tf.reduce_sum(y_pred*y_true)
    dice_coef = (2.*intersection+K.epsilon()) / (K.sum(y_pred)+K.sum(y_true)+K.epsilon())

    dice_loss = tf.reduce_mean(1-dice_coef)
    return dice_loss

def gen_dice(y_true, y_pred, eps=1e-6):
    """both tensors are [b, h, w, classes] and y_pred is in logit form"""

    # [b, h, w, classes]
    pred_tensor = tf.nn.softmax(y_pred)
    y_true_shape = tf.shape(y_true)

    # [b, h*w, classes]
    y_true = tf.reshape(y_true, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])
    y_pred = tf.reshape(pred_tensor, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])

    # [b, classes]
    # count how many of each class are present in
    # each image, if there are zero, then assign
    # them a fixed weight of eps
    counts = tf.reduce_sum(y_true, axis=1)
    weights = 1. / (counts)
    weights = tf.where(tf.math.is_finite(weights), weights, eps)

    multed = tf.reduce_sum(y_true * y_pred, axis=1)
    summed = tf.reduce_sum(y_true + y_pred, axis=1)

    # [b]
    numerators = tf.reduce_sum(weights*multed, axis=-1)
    denom = tf.reduce_sum(weights*summed, axis=-1)
    dices = 1. - 2. * numerators / denom
    dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
    return tf.reduce_mean(dices)

def Resblock_bn_DRU(input_tensor, channels, weight_decay = None):

    conv1 = Conv2D(filters=channels,
                   kernel_size=(3,3),
                   kernel_initializer="he_normal",
                   padding="same"
                   )(input_tensor)
    batchnorm1 = BatchNormalization()(conv1)
    active1 = Activation(activation="relu")(batchnorm1)

    conv2 = Conv2D(filters = 2*channels,
                   kernel_size=(3,3),
                   kernel_initializer="he_normal",
                   padding="same"
                   )(active1)
    batchnorm2 = BatchNormalization()(conv2)
    active2 = Activation(activation="relu")(batchnorm2)

    conv3 = Conv2D(filters=2*channels,
                   kernel_size=(1,1),
                   kernel_initializer="he_normal",
                   padding="same"
                   )(active2)
    batchnorm3 = BatchNormalization()(conv3)
    residual = Add()([input_tensor, batchnorm3])
    active3 = Activation(activation="relu")(residual)

    return active3

class DeepResUNet :
    """
    CODE INFO
    """

    def __init__(self, input_size, lr, num_classes):
        self.input_size = input_size
        self.lr = lr
        self.num_classes = num_classes
        #self.weight_decay = tf.keras.regularizers.L1L2(weight_decay)

    def build_net(self):

        input = tf.keras.Input(self.input_size)

        # Encoder - Input parts : 5x5 - Pool2
        En_conv5x5 = Conv2D(filters=128,
                            kernel_size=(5,5),
                            kernel_initializer="he_normal",
                            padding="same"
                            )(input)
        En_Conv5x5_bn = BatchNormalization()(En_conv5x5)
        En_Max2x2_1 = MaxPool2D(pool_size=(2,2),
                                strides=(2,2)
                                )(En_Conv5x5_bn)
        # block1 : Resblock x 2, Pool2
        En_rb1 = Resblock_bn_DRU(En_Max2x2_1,
                                  64)
        En_rb2 = Resblock_bn_DRU(En_rb1,
                                  64)
        En_add1 = Add()([En_Max2x2_1, En_rb2])
        En_pool1 = MaxPool2D(pool_size=(2,2),
                             strides=(2,2)
                             )(En_add1)
        # block2 : Resblock x 2, Pool2
        En_rb3 = Resblock_bn_DRU(En_pool1,
                                  64)
        En_rb4 = Resblock_bn_DRU(En_rb3,
                                  64)
        En_add2 = Add()([En_rb4, En_pool1])
        En_pool2 = MaxPool2D(pool_size=(2,2),
                             strides=(2,2)
                             )(En_add2)
        # block3 : Resblock x 2, Pool2
        En_rb5 = Resblock_bn_DRU(En_pool2,
                                  64)
        En_rb6 = Resblock_bn_DRU(En_rb5,
                                  64)
        En_add3 = Add()([En_rb6, En_pool2])
        En_pool3 = MaxPool2D(pool_size=(2,2),
                             strides=(2,2)
                             )(En_add3)
        # block4 : Resblock x 2
        En_rb7 = Resblock_bn_DRU(En_pool3,
                                  64)
        En_rb8 = Resblock_bn_DRU(En_rb7,
                                  64)
        En_add4 = Add()([En_rb8, En_pool3])

        # Decoder
        # block5 : tc, concat, conv1, rb, rb
        De_up1 = Conv2DTranspose(
            filters=128,
            kernel_size=(2,2),
            strides=(2,2)
        )(En_add4)
        De_concat1 = Concatenate()([De_up1, En_add3])
        De_conv1x1_1 = Conv2D(
            filters=128,
            kernel_size=(1,1),
            strides=(1,1),
            kernel_initializer="he_normal"
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
            kernel_size=(2,2),
            strides=(2,2)
        )(De_rb2)
        De_concat2 = Concatenate()([De_up2, En_add2])
        De_conv1x1_2 = Conv2D(
            filters=128,
            kernel_size=(1,1),
            strides=(1,1),
            kernel_initializer="he_normal"
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
            strides=(2, 2)
        )(De_rb4)
        De_concat3 = Concatenate()([De_up3, En_add1])
        De_conv1x1_3 = Conv2D(
            filters=128,
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer="he_normal"
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
            strides=(2, 2)
        )(De_rb6)
        De_concat4 = Concatenate()([De_up4, En_conv5x5])
        De_conv1x1_4 = Conv2D(
            filters=128,
            kernel_size=(1, 1),
            strides=(1, 1),
            kernel_initializer="he_normal"
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
            kernel_size=(1,1),
            strides=(1,1)
        )(De_rb8)

        De_last = Activation(activation="softmax")(De_conv1x1_5)
        #normalize_activation = Lambda(lambda x : x / tf.reduce_sum(x, len(x.get_shape())-1, True), name="normalize_active")(De_last)
        clip_activation = Lambda(lambda x : tf.clip_by_value(x, 1e-4, 1.-1e-4))(De_last)
        log_activation = Lambda(lambda x : K.log(x))(clip_activation)

        weight_map_ip = tf.keras.Input(shape=self.input_size[:2] + (self.num_classes,))
        weighted_softmax = multiply([log_activation, weight_map_ip])

        model = tf.keras.Model(inputs=(input, weight_map_ip), outputs=weighted_softmax)
        #model = keras.Model(inputs=input, outputs=log_activation)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss=cust_loss,
            metrics=UpdatedMeanIoU(num_classes=self.num_classes)
        )

        return model

if __name__ == "__main__" :

    INPUT_SHAPE = (512, 512, 3)
    NUM_CLASSES = 2
    LR = 1e-4
    BATCH_SIZE = 6
    seed=10

    # train
    image_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
    )
    mask_datagen_main = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
    )
    weight_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
    )
    image_generator = image_datagen.flow_from_directory(
       directory="./swham/train/images",
       class_mode=None,
       seed=seed,
       shuffle=True,
       target_size=(512, 512),
       batch_size=BATCH_SIZE
    )
    mask_generator = mask_datagen_main.flow_from_directory(
       directory="./swham/train/annotations",
       class_mode=None,
       seed=seed,
       shuffle=True,
       target_size=(512, 512),
       color_mode="grayscale",
       batch_size=BATCH_SIZE
    )
    weight_generator = weight_datagen.flow_from_directory(
        directory="./swham/train/weights_normal"
                  "",
        class_mode=None,
        seed=seed,
        shuffle=True,
        target_size=(512, 512),
        color_mode="grayscale",
        batch_size=BATCH_SIZE
    )

    # valid
    valid_img_datagen = ImageDataGenerator()
    valid_msk_datagen = ImageDataGenerator()
    valid_weight_datagen = ImageDataGenerator()
    valid_img_generator = valid_img_datagen.flow_from_directory(
        directory="./swham/val/images",
        target_size=(512, 512),
        seed=seed,
        shuffle=True,
        class_mode=None,
        batch_size=BATCH_SIZE
    )
    valid_mask_generator = valid_msk_datagen.flow_from_directory(
        directory="./swham/val/annotations",
        target_size=(512, 512),
        seed=seed,
        shuffle=True,
        class_mode=None,
        color_mode="grayscale",
        batch_size=BATCH_SIZE
    )
    valid_weight_generator = valid_weight_datagen.flow_from_directory(
        directory="./swham/val/weights",
        target_size=(512, 512),
        seed=seed,
        shuffle=True,
        class_mode=None,
        color_mode="grayscale",
        batch_size=BATCH_SIZE
    )

    def create_train_generator(img, weight, label):
        while True:
            for x1, x2, x3 in zip(img, weight, label):
                yield [x1, x2], x3

    train_gen = create_train_generator(image_generator, weight_generator, mask_generator)
    valid_gen = create_train_generator(valid_img_generator, valid_weight_generator, valid_mask_generator)

    DeepResUNet = DeepResUNet(input_size=INPUT_SHAPE, lr=LR, num_classes=NUM_CLASSES)
    model = DeepResUNet.build_net()
    # model.load_weights("./nw_weights/DRU_60.hdf5")
    model.summary()

    # # visualization
    # file_writer_img = tf.summary.create_file_writer('./nw_visual')
    # val_to_visualize = next(valid_gen)
    # trn_to_visualize = next(train_gen)
    #
    # with file_writer_img.as_default():
    #     tf.summary.image("TRN_IMAGE", trn_to_visualize[0][0]/255, step=0, max_outputs=BATCH_SIZE)
    #     tf.summary.image("TRN_LABEL", trn_to_visualize[1], step=0, max_outputs=BATCH_SIZE)
    #     tf.summary.image("TRN_WEIGHT", trn_to_visualize[0][1], step=0, max_outputs=BATCH_SIZE)
    #     tf.summary.image("VAL_IMAGE", val_to_visualize[0][0]/255, step=0, max_outputs=BATCH_SIZE)
    #     tf.summary.image("VAL_LABEL", val_to_visualize[1], step=0, max_outputs=BATCH_SIZE)
    #
    # def log_val_img(epoch, logs):
    #     val_pred = model.predict(val_to_visualize[0])
    #     trn_pred = model.predict(trn_to_visualize[0])
    #     with file_writer_img.as_default():
    #         tf.summary.image("VAL_PRED", val_pred, step=epoch, max_outputs=BATCH_SIZE)
    #         tf.summary.image("TRN_PRED", trn_pred, step=epoch, max_outputs=BATCH_SIZE)
    #
    # VISUALIZE = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_val_img)
    #
    # callbacks = [
    #     tf.keras.callbacks.ModelCheckpoint(filepath="./nw_weights/DRU_{epoch:02d}.hdf5"),
    #     tf.keras.callbacks.TensorBoard(log_dir="./nw_logs", update_freq="batch"),
    #     VISUALIZE
    #     #keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    # ]
    #
    # model.fit(
    #     train_gen,
    #     initial_epoch=60,
    #     validation_data=valid_gen,
    #     validation_batch_size=BATCH_SIZE,
    #     validation_steps=valid_img_generator.samples/BATCH_SIZE,
    #     steps_per_epoch=mask_generator.samples/BATCH_SIZE,
    #     epochs=100,
    #     callbacks=[callbacks]
    # )