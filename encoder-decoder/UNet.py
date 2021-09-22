import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import MeanIoU
#from losses_cus import DiceLoss, dice_coef, dice_loss
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K

# Optionset
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
session = tf.compat.v1.InteractiveSession(config=config)

def dice_loss(y_true, y_pred):
    dice_loss = 1-dice_coef(y_true, y_pred)
    return dice_loss

def dice_coef(y_true, y_pred):

    y_pred_f = y_true#K.flatten(y_pred)
    y_true_f = y_pred#K.flatten(y_true)

    intersection = K.sum(y_pred_f*y_true_f)

    return (2.*intersection+K.epsilon())/(K.sum(y_pred_f) + K.sum(y_true_f) + K.epsilon())

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
    weights = 1. / (counts)**2
    weights = tf.where(tf.math.is_finite(weights), weights, eps)

    multed = tf.reduce_sum(y_true * y_pred, axis=1)
    summed = tf.reduce_sum(y_true + y_pred, axis=1)

    # [b]
    numerators = tf.reduce_sum(weights*multed, axis=-1)
    denom = tf.reduce_sum(weights*summed, axis=-1)
    dices = 1. - 2. * numerators / denom
    dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
    return tf.reduce_mean(dices)

def gen_miou(y_true, y_pred, eps=1e-6):
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
    summed = tf.reduce_sum(y_true + y_pred, axis=1) - multed

    # [b]
    numerators = tf.reduce_sum(weights*multed, axis=-1)
    denom = tf.reduce_sum(weights*summed, axis=-1)
    ious = numerators / denom
    ious = tf.where(tf.math.is_finite(ious), ious, tf.zeros_like(ious))
    return tf.reduce_mean(ious)

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

class UNet :

    def __init__(self, input_shape, num_classes, lr) :
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lr = lr

    def build_net(self):
        input = keras.Input(self.input_shape)

        # Contracting path
        conv1 = Conv2D(filters=64, kernel_size=(3,3), padding="same",
                       strides=(1,1), activation="relu", name="conv1_contract")(input)
        conv2 = Conv2D(filters=64, kernel_size=(3,3), padding="same",
                       strides=(1,1), name="conv2_contract")(conv1)
        conv2 = Activation(activation="relu")(conv2)
        # /2 x /2 x 64
        pool1 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same",
                         name="pool1_contract")(conv2)

        conv3 = Conv2D(filters=128, kernel_size=(3,3), padding="same",
                       strides=(1,1), activation="relu", name="conv3_contract")(pool1)
        conv4 = Conv2D(filters=128, kernel_size=(3,3), padding="same",
                       strides=(1,1), activation="relu", name="conv4_contract")(conv3)
        # /4 x /4 x 128
        pool2 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same",
                          name="pool2_conctract")(conv4)

        conv5 = Conv2D(filters=256, kernel_size=(3,3), padding="same",
                       strides=(1,1), activation="relu", name="conv5_contract")(pool2)
        conv6 = Conv2D(filters=256, kernel_size=(3,3), padding="same",
                       strides=(1,1), activation="relu", name="conv6_contract")(conv5)
        # /8 x /8 x 256
        pool3 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same",
                          name="pool3_contract")(conv6)

        conv7 = Conv2D(filters=512, kernel_size=(3,3), padding="same",
                       strides=(1,1), activation="relu", name="conv7_contract")(pool3)
        conv8 = Conv2D(filters=512, kernel_size=(3,3), padding="same",
                       strides=(1,1), activation="relu", name="conv8_contract")(conv7)
        # /16 x /16 x 512
        pool4 = MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same",
                          name="pool4_contract")(conv8)

        # /16 x /16 x 1024
        conv9 = Conv2D(filters=1024, kernel_size=(3,3), padding="same",
                       strides=(1,1), activation="relu", name="conv9_contract")(pool4)
        conv10 = Conv2D(filters=1024, kernel_size=(3,3), padding="same",
                       strides=(1,1), activation="relu", name="conv10_contract")(conv9)
        conv10 = Dropout(rate=0.5)(conv10)

        # /8 x /8 x 512:512
        t_conv1 = Conv2DTranspose(filters=512, kernel_size=(2,2), padding="same",
                                  strides=(2,2), activation="relu", name="t_conv1_expand")(conv10)

        # expanding path
        concat1 = Concatenate()([conv8, t_conv1])
        conv1_r = Conv2D(filters=512, kernel_size=(3,3), padding="same",
                         strides=(1,1), activation="relu", name="conv1_r_expand")(concat1)
        conv2_r = Conv2D(filters=512, kernel_size=(3,3), padding="same",
                         strides=(1,1), activation="relu", name="conv2_r_expand")(conv1_r)
        # /4 x /4 x 256:256
        t_conv2 = Conv2DTranspose(filters=256, kernel_size=(2,2), padding="same",
                                  strides=(2,2), activation="relu", name="t_conv2_expand")(conv2_r)

        concat2 = Concatenate()([conv6, t_conv2])
        conv3_r = Conv2D(filters=256, kernel_size=(3, 3), padding="same",
                         strides=(1, 1), activation="relu", name="conv3_r_expand")(concat2)
        conv4_r = Conv2D(filters=256, kernel_size=(3, 3), padding="same",
                         strides=(1, 1), activation="relu", name="conv4_r_expand")(conv3_r)
        # /2 x /2 x 128:128
        t_conv3 = Conv2DTranspose(filters=128, kernel_size=(2,2), padding="same",
                                  strides=(2,2), activation="relu", name="t_conv3_expand")(conv4_r)

        concat3 = Concatenate()([conv4, t_conv3])
        conv5_r = Conv2D(filters=128, kernel_size=(3, 3), padding="same",
                         strides=(1, 1), activation="relu", name="conv5_r_expand")(concat3)
        conv6_r = Conv2D(filters=128, kernel_size=(3, 3), padding="same",
                         strides=(1, 1), activation="relu", name="conv6_r_expand")(conv5_r)
        # /1 x /1 x 64:64
        t_conv4 = Conv2DTranspose(filters=64, kernel_size=(2,2), padding="same",
                                  strides=(2,2), activation="relu", name="t_conv4_expand")(conv6_r)

        concat4 = Concatenate()([conv2, t_conv4])
        conv7_r = Conv2D(filters=64, kernel_size=(3, 3), padding="same",
                         strides=(1, 1), activation="relu", name="conv7_r_expand")(concat4)
        conv8_r = Conv2D(filters=64, kernel_size=(3, 3), padding="same",
                         strides=(1, 1), activation="relu", name="conv8_r_expand")(conv7_r)

        # output
        conv9_r = Conv2D(filters=self.num_classes, kernel_size=(3, 3), padding="same",
                         strides=(1, 1), activation="sigmoid", name="output")(conv8_r)
        model = keras.Model(inputs=input, outputs=conv9_r)
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=gen_dice,
            metrics=UpdatedMeanIoU(num_classes=self.num_classes)
        )

        return model

if __name__ == "__main__" :

    SEED=10
    BATCH_SIZE=1
    INPUT_SHAPE = (512,512,3)

    image_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90
    )
    mask_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90
    )

    image_generator = image_datagen.flow_from_directory(
        #directory="F:\\Data_list\\master_paper\\cultural_heritage\\deep_learning\\U-net_Git2\\Dataset\\train\\image",
        directory="E:\\Deep_learning_dataset\\Cityscapes\\dataset\\dataset_mod\\train\\images",
        class_mode=None,
        seed=SEED,
        shuffle=True,
        target_size=(512, 512),
        batch_size=BATCH_SIZE
    )
    mask_generator = mask_datagen.flow_from_directory(
        #directory="F:\\Data_list\\master_paper\\cultural_heritage\\deep_learning\\U-net_Git2\\Dataset\\train\\label_sl",
        directory="E:\\Deep_learning_dataset\\Cityscapes\\dataset\\dataset_mod\\train\\labels",
        class_mode=None,
        seed=SEED,
        shuffle=True,
        target_size=(512, 512),
        color_mode="grayscale",
        batch_size=BATCH_SIZE
    )

    train_gen = zip(image_generator, mask_generator)

    a = UNet(input_shape=(512, 512, 3), num_classes=2, lr=0.01)
    model = a.build_net()
    model.summary()

    # model.fit(
    #     train_gen,
    #     steps_per_epoch=image_generator.samples / BATCH_SIZE,
    #     epochs=10
    # )
