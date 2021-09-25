"""
Kangmin Park, lab for Sensor and Modeling, Univ. of Seoul.
Date : 2021.01.19.
This code is Dual Attention Network, based on Jun Fu, etal., <Dual Attention Network for Scene Segmentation>, 2018. V4.
"""
from utils import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import AvgPool2D
from tensorflow.python.keras import backend as K
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, ResNet101

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


def make_variables(k, initializer):
  return tf.Variable(initializer(shape=[k], dtype=tf.float32))

def spatial_attention_module(input_tensor) :

    tensor_shape = input_tensor.shape

    # "we first apply convolution layer to obtain features of dimension reduction."
    A = Conv2D(filters=tensor_shape[3]//8,
               kernel_size=(3,3),
               padding="same")(input_tensor)

    # "we first feed it into a convolution layers to generate two new feature maps B and C, respectively."
    B = Conv2D(filters=A.shape[3],
               kernel_size=(3,3),
               padding="same")(A)
    C = Conv2D(filters=A.shape[3],
               kernel_size=(3,3),
               padding="same")(A)
    D = Conv2D(filters=A.shape[3],
               kernel_size=(3,3),
               padding="same")(A)

    B = tf.reshape(B,
                   shape=[-1, A.shape[1]*A.shape[2], A.shape[3]])
    C = tf.reshape(C,
                   shape=[-1, A.shape[1]*A.shape[2], A.shape[3]])
    D = tf.reshape(D,
                   shape=[-1, A.shape[1]*A.shape[2], A.shape[3]])

    # Sptial attention mask : C shape : (Batch, N(HXW), C), B shape : (Batch, N, C)
    _S = tf.matmul(C, tf.transpose(B, [0, 2, 1]))
    S = keras.activations.softmax(_S, axis=2)#tf.exp(_S) / tf.reduce_sum(tf.exp(_S), axis=2)

    attention_score = tf.matmul(tf.transpose(D, [0, 2, 1]), tf.transpose(S, [0, 2, 1]))
    attention_score = tf.reshape(attention_score, shape=[-1, A.shape[1], A.shape[2], A.shape[3]])
    # tf.Variable(tf.zeros_initializer(shape=1)) #TypeError: object() takes no parameters
    scale = make_variables(1, tf.zeros_initializer())
    E_spatial = scale*attention_score+A

    return E_spatial

def channel_attention_module(input_tensor):

    tensor_shape = input_tensor.shape

    A = Conv2D(filters=tensor_shape[3]//8,
               kernel_size=(3,3),
               padding="same")(input_tensor)

    A_1 = tf.reshape(A,
                     shape=[-1, A.shape[1]*A.shape[2], A.shape[3]])
    A_2 = tf.reshape(A,
                     shape=[-1, A.shape[1]*A.shape[2], A.shape[3]])
    A_3 = tf.reshape(A,
                     shape=[-1, A.shape[1]*A.shape[2], A.shape[3]])

    # Channel attention mask : shape of A_1~3 : [Batch, N(HxW), C]
    _X = tf.matmul(tf.transpose(A_1, [0, 2, 1]), A_2)
    X = keras.activations.softmax(_X, axis=2)#tf.exp(_X) / tf.reduce_sum(tf.exp(_X), axis=2)

    attention_score = tf.matmul(A_3, tf.transpose(X, [0, 2, 1]))
    attention_score = tf.reshape(attention_score, shape=[-1, A.shape[1], A.shape[2], A.shape[3]])
    scale = make_variables(1, tf.zeros_initializer())
    E_channel = scale*attention_score+A

    return E_channel


class DAN_ResNet50 :

    def __init__(self,
                 input_shape,
                 num_classes,
                 learning_rate=0.01):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lr = learning_rate

    def build_net(self):

        input = keras.Input(self.input_shape)

        # Encoder Dilated-ResNet 50 (Not Pretrained)
        en_conv1 = Conv2D(
            filters=64,
            kernel_size=7,
            strides=2,
            padding="same",
            activation="relu",
            name="en_conv1"
        )(input)
        en_pool1 = MaxPool2D(
            pool_size=3,
            strides=2,
            padding="same",
            name="en_pool1"
        )(en_conv1)

        # en_rb1~3 : 1/4
        en_rb1 = Resblock_bn(input=en_pool1,
                             knum_in=64,
                             knum_out=256,
                             layer_name="en_rb1")
        en_rb2 = Resblock_bn(input=en_rb1,
                             knum_in=64,
                             knum_out=256,
                             layer_name="en_rb2")
        en_rb3 = Resblock_bn(input=en_rb2,
                             knum_in=64,
                             knum_out=256,
                             layer_name="en_rb3")

        # en_rb4~7 / 이후까지 다 : 1/8
        en_rb4 = Resblock_bn(input=en_rb3,
                             knum_in=128,
                             knum_out=512,
                             layer_name="en_rb4",
                             verbose=True)
        en_rb5 = Resblock_bn(input=en_rb4,
                             knum_in=128,
                             knum_out=512,
                             layer_name="en_rb5")
        en_rb6 = Resblock_bn(input=en_rb5,
                             knum_in=128,
                             knum_out=512,
                             layer_name="en_rb6")
        en_rb7 = Resblock_bn(input=en_rb6,
                             knum_in=128,
                             knum_out=512,
                             layer_name="en_rb7")

        # Dilated Resblock part1 without downsampling
        en_rb8 = Resblock_bn(input=en_rb7,
                             knum_in=256,
                             knum_out=1024,
                             layer_name="en_rb8",
                             dilation=(3,3))
        en_rb9 = Resblock_bn(input=en_rb8,
                             knum_in=256,
                             knum_out=1024,
                             layer_name="en_rb9",
                             dilation=(3,3))
        en_rb10 = Resblock_bn(input=en_rb9,
                             knum_in=256,
                             knum_out=1024,
                             layer_name="en_rb10",
                             dilation=(3,3))
        en_rb11 = Resblock_bn(input=en_rb10,
                             knum_in=256,
                             knum_out=1024,
                             layer_name="en_rb11",
                             dilation=(3,3))
        en_rb12 = Resblock_bn(input=en_rb11,
                             knum_in=256,
                             knum_out=1024,
                             layer_name="en_rb12",
                             dilation=(3,3))
        en_rb13 = Resblock_bn(input=en_rb12,
                             knum_in=256,
                             knum_out=1024,
                             layer_name="en_rb13",
                             dilation=(3,3))

        # Dilated Resblock part2 without downsampling
        en_rb14 = Resblock_bn(input=en_rb13,
                              knum_in=512,
                              knum_out=2048,
                              layer_name="en_rb14",
                              dilation=(3,3))
        en_rb15 = Resblock_bn(input=en_rb14,
                              knum_in=512,
                              knum_out=2048,
                              layer_name="en_rb15",
                              dilation=(3,3))
        en_rb16 = Resblock_bn(input=en_rb15,
                              knum_in=512,
                              knum_out=2048,
                              layer_name="en_rb16",
                              dilation=(3,3))
        en_rb17 = Resblock_bn(input=en_rb16,
                              knum_in=512,
                              knum_out=2048,
                              layer_name="en_rb17",
                              dilation=(3,3))

        # Spatial Attention Map
        E_spatial = spatial_attention_module(en_rb17)
        E_spatial = Conv2D(filters=E_spatial.shape[3],
                           padding="same",
                           kernel_size=3,
                           kernel_initializer="he_normal",
                           name="conv_e_spatial_1")(E_spatial)
        E_spatial = BatchNormalization()(E_spatial)
        E_spatial = Activation(activation='relu')(E_spatial)
        E_spatial = Conv2D(filters=E_spatial.shape[3],
                           padding="same",
                           kernel_size=3,
                           kernel_initializer="he_normal",
                           name="conv_e_spatial_2")(E_spatial)

        # Channel Attention Map
        E_channel = channel_attention_module(en_rb17)
        E_channel = Conv2D(filters=E_channel.shape[3],
                           padding="same",
                           kernel_size=3,
                           kernel_initializer="he_normal",
                           name="conv_e_channel_1")(E_channel)
        E_channel = BatchNormalization()(E_channel)
        E_channel = Activation(activation='relu')(E_channel)
        E_channel = Conv2D(filters=E_spatial.shape[3],
                           padding="same",
                           kernel_size=3,
                           kernel_initializer="he_normal",
                           name="conv_e_channel_2")(E_channel)

        # 1/4
        sum_fusion = tf.add(E_spatial, E_channel)
        de_conv1 = Conv2DTranspose(filters=128,
                                   kernel_size=3,
                                   strides=2,
                                   padding="same",
                                   activation="relu",
                                   kernel_initializer="he_normal")(sum_fusion)
        de_conv1_conc = Concatenate()([de_conv1, en_rb3])
        de_conv1_conv = Conv2D(filters=128,
                               kernel_size=3,
                               padding="same",
                               kernel_initializer="he_normal")(de_conv1_conc)
        de_conv1_batch = BatchNormalization()(de_conv1_conv)
        de_conv1_activ = Activation(activation="relu")(de_conv1_batch)

        # 1/2
        de_conv2 = Conv2DTranspose(filters=64,
                                   kernel_size=2,
                                   strides=2,
                                   padding="same",
                                   activation="relu",
                                   kernel_initializer="he_normal")(de_conv1_activ)
        de_conv2_conc = Concatenate()([de_conv2, en_conv1])
        de_conv2_conv = Conv2D(filters=64,
                               kernel_size=3,
                               padding="same",
                               kernel_initializer="he_normal")(de_conv2_conc)
        de_conv2_batch = BatchNormalization()(de_conv2_conv)
        de_conv2_activ = Activation(activation="relu")(de_conv2_batch)

        # original
        predict_map = Conv2DTranspose(
            filters=self.num_classes,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="softmax",
            name="t_conv_predict",
        )(de_conv2_activ)

        model = keras.Model(inputs=input, outputs=predict_map)
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=SparseCategoricalCrossentropy(),
            metrics=UpdatedMeanIoU(num_classes=self.num_classes),
        )
        return model

if __name__ == "__main__" :

    INPUT_SHAPE = (512, 512, 3)
    NUM_CLASSES = 2
    LR = 1e-4
    BATCH_SIZE=4
    seed=10

    # image_datagen = ImageDataGenerator(
    #     horizontal_flip=True,
    #     vertical_flip=True,
    # )
    # mask_datagen_main = ImageDataGenerator(
    #     horizontal_flip=True,
    #     vertical_flip=True,
    # )
    #
    # image_generator = image_datagen.flow_from_directory(
    #    directory="./swham/train/images",
    #    class_mode=None,
    #    seed=seed,
    #    shuffle=True,
    #    target_size=(512, 512),
    #    batch_size=BATCH_SIZE
    # )
    # mask_generator = mask_datagen_main.flow_from_directory(
    #     directory="./swham/train/annotations",
    #     class_mode=None,
    #     seed=seed,
    #     shuffle=True,
    #     target_size=(512, 512),
    #     color_mode="grayscale",
    #     batch_size=BATCH_SIZE
    # )
    #
    # # valid
    # valid_img_datagen = ImageDataGenerator(
    #     horizontal_flip=True,
    #     vertical_flip=True,
    # )
    # valid_msk_datagen = ImageDataGenerator(
    #     horizontal_flip=True,
    #     vertical_flip=True,
    # )
    # valid_img_generator = valid_img_datagen.flow_from_directory(
    #     directory="./swham/val/images",
    #     target_size=(512, 512),
    #     seed=seed,
    #     shuffle=True,

    #     class_mode=None,
    #     batch_size=BATCH_SIZE
    # )
    # valid_mask_generator = valid_msk_datagen.flow_from_directory(
    #     directory="./swham/val/annotations",
    #     target_size=(512, 512),
    #     seed=seed,
    #     shuffle=True,
    #     class_mode=None,
    #     color_mode="grayscale",
    #     batch_size=BATCH_SIZE
    # )
    #
    # def create_train_generator(img, label):
    #     while True:
    #         for x1, x2 in zip(img, label):
    #             yield x1, x2
    #
    # train_gen = create_train_generator(image_generator, mask_generator)
    # valid_gen = create_train_generator(valid_img_generator, valid_mask_generator)

    DualNet = DAN_ResNet50(input_shape=INPUT_SHAPE,
                           num_classes=NUM_CLASSES,
                           learning_rate=LR)
    model = DualNet.build_net()
    model.summary()

    # callbacks = [
    #     tf.keras.callbacks.ModelCheckpoint(filepath="./DAN_weights/DAN_{epoch:02d}.hdf5"),
    #     tf.keras.callbacks.TensorBoard(log_dir="./DAN_logs", update_freq="batch")
    # ]
    #
    # model.fit(
    #     train_gen,
    #     validation_data=valid_gen,
    #     validation_batch_size=BATCH_SIZE,
    #     validation_steps=valid_img_generator.samples/BATCH_SIZE,
    #     steps_per_epoch=mask_generator.samples/BATCH_SIZE,
    #     epochs=100,
    #     callbacks=[callbacks]
    # )
