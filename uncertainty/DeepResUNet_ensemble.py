import os
import numpy as np
from tifffile import tifffile as tifi
import tensorflow as tf
from PIL import Image
from tensorflow import keras
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
    tifi.imwrite("./data_for_sangam_pred/dist5_pred/"+ name+".tif", mask[:, :, 1])

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

    def __init__(self):

        return

    def build_net(self,
                  input_shape,
                  num_classes=2):

        input = tf.keras.Input(input_shape)

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
            filters=num_classes,
            kernel_size=(1,1),
            strides=(1,1)
        )(De_rb8)

        De_last = Activation(activation="softmax")(De_conv1x1_5)

        model = keras.Model(inputs=input,
                            outputs=De_last)

        return model

class Ensemble :

    def __init__(self,
                 input_shape,
                 lr,
                 num_classes):
        self.input_shape = input_shape
        self.lr = lr
        self.num_classes = num_classes

    def build_ensemble(self):

        dru_inst = DeepResUNet()
        dru1 = dru_inst.build_net(input_shape=self.input_shape,
                                  num_classes=self.num_classes)
        dru2 = dru_inst.build_net(input_shape=self.input_shape,
                                  num_classes=self.num_classes)
        dru3 = dru_inst.build_net(input_shape=self.input_shape,
                                  num_classes=self.num_classes)
        dru4 = dru_inst.build_net(input_shape=self.input_shape,
                                  num_classes=self.num_classes)
        dru5 = dru_inst.build_net(input_shape=self.input_shape,
                                  num_classes=self.num_classes)

        input = keras.Input(shape=self.input_shape)

        dru1 = dru1(input)
        dru2 = dru2(input)
        dru3 = dru3(input)
        dru4 = dru4(input)
        dru5 = dru5(input)

        dru_ensemble = Add()([dru1, dru2, dru3, dru4, dru5])/5

        model = tf.keras.Model(inputs=input, outputs=dru_ensemble)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss=SparseCategoricalCrossentropy(),
            metrics=UpdatedMeanIoU(num_classes=self.num_classes)
        )

        return model

if __name__ == "__main__" :

    INPUT_SHAPE=(512, 512, 3)
    NUM_CLASSES=2
    # start_lr = 0.0001
    # end_lr = 0.000001
    # decay_step = 100
    # LR = tf.keras.optimizers.schedules.PolynomialDecay(
    #     start_lr,
    #     decay_steps=decay_step,
    #     end_learning_rate=end_lr,
    #     power=0.5
    # )
    LR = 0.0001
    BATCH_SIZE = 1
    seed = 10

    # train
    image_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
    )
    mask_datagen_main = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
    )

    image_generator = image_datagen.flow_from_directory(
        directory="./renewal_dataset_0324/train/images",
        class_mode=None,
        seed=seed,
        shuffle=True,
        target_size=(512, 512),
        batch_size=BATCH_SIZE
    )
    mask_generator = mask_datagen_main.flow_from_directory(
        directory="./renewal_dataset_0324/train/labels",
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


    def create_train_generator(img1, label):
        while True:
            for x1, x2,  in zip(img1, label):
                yield x1, x2

    train_gen = create_train_generator(image_generator, mask_generator)
    valid_gen = create_train_generator(valid_img_generator, valid_mask_generator)


    # pred_img_datagen=ImageDataGenerator()
    # pred_img_generator = pred_img_datagen.flow_from_directory(
    #     directory="./data_for_sangam_pred/dist5_clipped",
    #     target_size=(512, 512),
    #     batch_size=1,
    #     shuffle=False,
    #     class_mode=None
    # )

    ensemble_model = Ensemble(input_shape=INPUT_SHAPE,
                              lr=LR,
                              num_classes=NUM_CLASSES)
    model = ensemble_model.build_ensemble()
    # model.load_weights("./ensemble/naive/m5/dru_ensemble_m5_47.hdf5")
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath="./ensemble/naive/ensemble/dru_ensemble_{epoch:02d}.hdf5"),
        # tf.keras.callbacks.TensorBoard(log_dir="./DRU_CA_rgb_callbacks/finetune_logs", update_freq="batch"),
        # VISUALIZE
        # keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    ]

    model.fit(
        train_gen,
        validation_data=valid_gen,
        validation_batch_size=BATCH_SIZE,
        validation_steps=valid_img_generator.samples / BATCH_SIZE,
        steps_per_epoch= valid_mask_generator.samples / BATCH_SIZE,
        epochs=50,
        callbacks=[callbacks]
    )

    # for data, name in zip(pred_img_generator, pred_img_generator.filenames) :
    #     name = str(name).split('/')[-1].split('\\')[-1].split('.')[0]
    #     landcover_test = model.predict(
    #        data,
    #        batch_size=1,
    #        verbose=1,
    #        steps=pred_img_generator.samples/1.
    #     )
    #     class2color(landcover_test, name)