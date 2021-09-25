import os
import numpy as np
from PIL import Image
from tifffile import tifffile as tifi
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

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

def spatial_attention_module(input_tensor) :

    tensor_shape = input_tensor.shape

    # "we first apply convolution layer to obtain features of dimension reduction."
    A = Conv2D(filters=tensor_shape[3]//8,
               kernel_size=(3,3),
               padding="same",
               trainable=False)(input_tensor)

    # "we first feed it into a convolution layers to generate two new feature maps B and C, respectively."
    B = Conv2D(filters=A.shape[3],
               kernel_size=(3,3),
               padding="same",
               trainable=False)(A)
    C = Conv2D(filters=A.shape[3],
               kernel_size=(3,3),
               padding="same",
               trainable=False)(A)
    D = Conv2D(filters=A.shape[3],
               kernel_size=(3,3),
               padding="same",
               trainable=False)(A)

    B = tf.reshape(B,
                   shape=[-1, A.shape[1]*A.shape[2], A.shape[3]])
    C = tf.reshape(C,
                   shape=[-1, A.shape[1]*A.shape[2], A.shape[3]])
    D = tf.reshape(D,
                   shape=[-1, A.shape[1]*A.shape[2], A.shape[3]])

    # Sptial attention mask : C shape : (Batch, N(HXW), C), B shape : (Batch, N, C)
    _S = tf.matmul(C, tf.transpose(B, [0, 2, 1]))
    S = tf.keras.activations.softmax(_S, axis=2)

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
               padding="same",
               trainable=False)(input_tensor)

    A_1 = tf.reshape(A,
                     shape=[-1, A.shape[1]*A.shape[2], A.shape[3]])
    A_2 = tf.reshape(A,
                     shape=[-1, A.shape[1]*A.shape[2], A.shape[3]])
    A_3 = tf.reshape(A,
                     shape=[-1, A.shape[1]*A.shape[2], A.shape[3]])

    # Channel attention mask : shape of A_1~3 : [Batch, N(HxW), C]
    _X = tf.matmul(tf.transpose(A_1, [0, 2, 1]), A_2)
    X = tf.keras.activations.softmax(_X, axis=2)

    attention_score = tf.matmul(A_3, tf.transpose(X, [0, 2, 1]))
    attention_score = tf.reshape(attention_score, shape=[-1, A.shape[1], A.shape[2], A.shape[3]])
    scale = make_variables(1, tf.zeros_initializer())
    E_channel = scale*attention_score+A

    return E_channel

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
    tifi.imwrite("./data_for_sangam_pred/dist1_pred/"+ name+".tif", mask[:, :, 1])

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

class SA_DeepResUNet :
    """
    CODE INFO
    """

    def __init__(self, input_shape, lr, num_classes, batch_size):
        self.input_shape = input_shape
        self.lr = lr
        self.num_classes = num_classes
        self.batch_size = batch_size
        #self.weight_decay = tf.keras.regularizers.L1L2(weight_decay)

    def pretrained(self):

        input = tf.keras.Input(self.input_shape)

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


        # Spatial Attention Map
        E_spatial = spatial_attention_module(En_add4)
        E_spatial = Conv2D(filters=E_spatial.shape[3],
                           padding="same",
                           kernel_size=3,
                           kernel_initializer="he_normal",
                           name="conv_e_spatial_1",
                           trainable=False)(E_spatial)
        E_spatial = BatchNormalization(trainable=False)(E_spatial)
        E_spatial = Activation(activation='relu')(E_spatial)
        E_spatial = Conv2D(filters=E_spatial.shape[3],
                           padding="same",
                           kernel_size=3,
                           kernel_initializer="he_normal",
                           name="conv_e_spatial_2",
                           trainable=False)(E_spatial)

        # Decoder
        # block5 : tc, concat, conv1, rb, rb
        De_up1 = Conv2DTranspose(
            filters=128,
            kernel_size=(2, 2),
            strides=(2, 2),
            trainable=False
        )(E_spatial)
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
        input = tf.keras.Input(self.input_shape)

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


        # Spatial Attention Map
        E_spatial = spatial_attention_module(En_add4)
        E_spatial = Conv2D(filters=E_spatial.shape[3],
                           padding="same",
                           kernel_size=3,
                           kernel_initializer="he_normal",
                           name="conv_e_spatial_1",
                           trainable=False)(E_spatial)
        E_spatial = BatchNormalization(trainable=False)(E_spatial)
        E_spatial = Activation(activation='relu')(E_spatial)
        E_spatial = Conv2D(filters=E_spatial.shape[3],
                           padding="same",
                           kernel_size=3,
                           kernel_initializer="he_normal",
                           name="conv_e_spatial_2",
                           trainable=False)(E_spatial)

        # Decoder
        # block5 : tc, concat, conv1, rb, rb
        De_up1 = Conv2DTranspose(
            filters=128,
            kernel_size=(2, 2),
            strides=(2, 2),
            trainable=False
        )(E_spatial)
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

if __name__ == "__main__":

    INPUT_SHAPE = (512, 512, 3)
    NUM_CLASSES = 2
    # start_lr = 0.0001
    # end_lr = 0.000001
    # decay_step = 100
    # LR = tf.keras.optimizers.schedules.PolynomialDecay(
    #     start_lr,
    #     decay_steps=decay_step,
    #     end_learning_rate=end_lr,
    #     power=0.5
    # )
    LR = 0.001
    BATCH_SIZE = 4
    seed = 10

    # # train
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
    #     directory="./renewal_dataset_0324/train/images",
    #     class_mode=None,
    #     seed=seed,
    #     shuffle=True,
    #     target_size=(512, 512),
    #     batch_size=BATCH_SIZE
    # )
    # mask_generator = mask_datagen_main.flow_from_directory(
    #     directory="./renewal_dataset_0324/train/labels",
    #     class_mode=None,
    #     seed=seed,
    #     shuffle=True,
    #     target_size=(512, 512),
    #     color_mode="grayscale",
    #     batch_size=BATCH_SIZE
    # )

    # # valid
    # valid_img_datagen = ImageDataGenerator()
    # valid_msk_datagen = ImageDataGenerator()
    #
    # valid_img_generator = valid_img_datagen.flow_from_directory(
    #     directory="./renewal_dataset_0324/valid/images",
    #     target_size=(512, 512),
    #     seed=seed,
    #     shuffle=True,
    #     class_mode=None,
    #     batch_size=BATCH_SIZE
    # )
    # valid_mask_generator = valid_msk_datagen.flow_from_directory(
    #     directory="./renewal_dataset_0324/valid/labels",
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

    # train_gen = create_train_generator(image_generator, mask_generator)
    # valid_gen = create_train_generator(valid_img_generator, valid_mask_generator)

    pred_img_datagen = ImageDataGenerator()

    pred_img_generator = pred_img_datagen.flow_from_directory(
        directory="./data_for_sangam_pred/dist5_clipped",
        target_size=(512, 512),
        batch_size=1,
        shuffle=False,
        class_mode=None
    )

    # # Pretrained model
    # pretrained = SA_DeepResUNet(input_shape=INPUT_SHAPE, lr=LR, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE).pretrained()
    # pretrained.load_weights("./DRU_SA_rgb_callbacks/finetune_weights/DRU_SA_rgb_49.hdf5")

    # Calibrate model
    SA_DRU = SA_DeepResUNet(input_shape=INPUT_SHAPE, lr=LR, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE)
    model = SA_DRU.build_net()
    model.load_weights("./DRU_SA_rgb_callbacks/calibrated_weights/DRU_SA_rgb_calib_50.hdf5")
    # for i in range(len(pretrained.layers) - 1):
    #     model.layers[i].set_weights(pretrained.layers[i].get_weights())
    model.summary()

    # # visualization
    # file_writer_img = tf.summary.create_file_writer('./renewal_dataset_0324_vis')
    # val_to_visualize = next(valid_gen)
    # trn_to_visualize = next(train_gen)
    #
    # with file_writer_img.as_default():
    #     tf.summary.image("TRN_IMAGE", trn_to_visualize[0]/255, step=0, max_outputs=BATCH_SIZE)
    #     tf.summary.image("TRN_LABEL", trn_to_visualize[1], step=0, max_outputs=BATCH_SIZE)
    #     tf.summary.image("VAL_IMAGE", val_to_visualize[0]/255, step=0, max_outputs=BATCH_SIZE)
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

    # callbacks = [
    #     tf.keras.callbacks.ModelCheckpoint(filepath="./DRU_SA_rgb_callbacks/calibrated_weights/DRU_SA_rgb_calib_{epoch:02d}.hdf5"),
    #     # tf.keras.callbacks.TensorBoard(log_dir="./DRU_SA_rgb_callbacks/finetune_logs", update_freq="batch"),
    #     # VISUALIZE
    #     # keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    # ]

    # model.fit(
    #     valid_gen,
    #     # initial_epoch=44,
    #     # validation_data=valid_gen,
    #     # validation_batch_size=BATCH_SIZE,
    #     # validation_steps=valid_img_generator.samples / BATCH_SIZE,
    #     steps_per_epoch=valid_mask_generator.samples / BATCH_SIZE,
    #     epochs=50,
    #     callbacks=[callbacks]
    # )

    for data, name in zip(pred_img_generator, pred_img_generator.filenames) :
        name = str(name).split('/')[-1].split('\\')[-1].split('.')[0]
        pred_test = model.predict(
           data,
           batch_size=1,
           verbose=1,
           steps=pred_img_generator.samples/1.
        )
        print(pred_test.shape)
        class2color(pred_test, name)