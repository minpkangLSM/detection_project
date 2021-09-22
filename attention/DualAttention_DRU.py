import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
               padding="same")(input_tensor)

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
    mask = np.argmax(mask, axis=-1)
    mask_zero = np.zeros([512, 512, 3])
    name = name+".png"
    for class_num in range(2):
        mask_label = mask == class_num
        if class_num == 0:
            mask_zero[mask_label] = [0, 0, 0]
        elif class_num == 1:
            mask_zero[mask_label] = [255, 255, 255]

    print(name)
    im = Image.fromarray(mask_zero.astype(np.uint8))
    im.save("./DA_DRU_rgbd_f_pred/"+name, 'png')

class DA_DeepResUNet :
    """
    CODE INFO
    """

    def __init__(self, input_shape, offground_shape, ground_shape, lr, num_classes):
        self.input_shape = input_shape
        self.offground_shape = offground_shape
        self.ground_shape = ground_shape
        self.lr = lr
        self.num_classes = num_classes
        #self.weight_decay = tf.keras.regularizers.L1L2(weight_decay)

    def build_net(self):

        input = tf.keras.Input(self.input_shape)
        offground = tf.keras.Input(self.offground_shape)
        ground = tf.keras.Input(self.ground_shape)
        ndsm = offground - ground
        rgbd = Concatenate()([input, ndsm])

        # Encoder - Input parts : 5x5 - Pool2
        En_conv5x5 = Conv2D(filters=128,
                            kernel_size=(5,5),
                            kernel_initializer="he_normal",
                            padding="same"
                            )(rgbd)
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


        # Spatial Attention Map
        E_spatial = spatial_attention_module(En_add4)
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
        E_channel = channel_attention_module(En_add4)
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

        # Decoder
        # block5 : tc, concat, conv1, rb, rb
        De_up1 = Conv2DTranspose(
            filters=128,
            kernel_size=(2,2),
            strides=(2,2)
        )(sum_fusion)
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

        model = tf.keras.Model(inputs=(input, offground, ground), outputs=De_last)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss=SparseCategoricalCrossentropy(),
            metrics=UpdatedMeanIoU(num_classes=self.num_classes)
        )

        return model

if __name__ == "__main__":

    INPUT_SHAPE = (512, 512, 3)
    OFFGROUND_SHAPE = (512, 512, 1)
    GROUND_SHAPE = (512, 512, 1)
    NUM_CLASSES = 2
    start_lr = 0.001
    end_lr = 0.00001
    decay_step = 100
    LR = tf.keras.optimizers.schedules.PolynomialDecay(
        start_lr,
        decay_steps=decay_step,
        end_learning_rate=end_lr,
        power=0.5
    )
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
    # offground_datagen_main = ImageDataGenerator(
    #     horizontal_flip=True,
    #     vertical_flip=True,
    # )
    # ground_datagen_main = ImageDataGenerator(
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
    # offground_generator = offground_datagen_main.flow_from_directory(
    #     directory="./renewal_dataset_0324/train/offground",
    #     class_mode=None,
    #     seed=seed,
    #     shuffle=True,
    #     target_size=(512, 512),
    #     color_mode="grayscale",
    #     batch_size=BATCH_SIZE
    # )
    # ground_generator = ground_datagen_main.flow_from_directory(
    #     directory="./renewal_dataset_0324/train/ground",
    #     class_mode=None,
    #     seed=seed,
    #     shuffle=True,
    #     target_size=(512, 512),
    #     color_mode="grayscale",
    #     batch_size=BATCH_SIZE
    # )
    #
    # # valid
    # valid_img_datagen = ImageDataGenerator()
    # valid_msk_datagen = ImageDataGenerator()
    # valid_offground_datagen = ImageDataGenerator()
    # valid_ground_datagen = ImageDataGenerator()
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
    # valid_offground_generator = valid_offground_datagen.flow_from_directory(
    #     directory="./renewal_dataset_0324/valid/offground",
    #     target_size=(512, 512),
    #     seed=seed,
    #     shuffle=True,
    #     class_mode=None,
    #     color_mode="grayscale",
    #     batch_size=BATCH_SIZE
    # )
    # valid_ground_generator = valid_ground_datagen.flow_from_directory(
    #     directory="./renewal_dataset_0324/valid/ground",
    #     target_size=(512, 512),
    #     seed=seed,
    #     shuffle=True,
    #     class_mode=None,
    #     color_mode="grayscale",
    #     batch_size=BATCH_SIZE
    # )
    #
    #
    # def create_train_generator(img, offground, ground, label):
    #     while True:
    #         for x1, x2, x3, x4 in zip(img, offground, ground, label):
    #             yield (x1, x2, x3), x4
    #
    #
    # train_gen = create_train_generator(image_generator, offground_generator, ground_generator, mask_generator)
    # valid_gen = create_train_generator(valid_img_generator, valid_offground_generator, valid_ground_generator,
    #                                    valid_mask_generator)

    pred_img_datagen = ImageDataGenerator()
    pred_ground_datagen = ImageDataGenerator()
    pred_offground_datagen = ImageDataGenerator()

    pred_img_generator = pred_img_datagen.flow_from_directory(
        directory="./renewal_dataset_0326_myeonmok/images",
        target_size=(512, 512),
        batch_size=1,
        shuffle=False,
        class_mode=None
    )
    pred_ground_generator = pred_ground_datagen.flow_from_directory(
        directory="./renewal_dataset_0326_myeonmok/grounds",
        target_size=(512, 512),
        batch_size=1,
        shuffle=False,
        color_mode="grayscale",
        class_mode=None
    )
    pred_offground_generator = pred_offground_datagen.flow_from_directory(
        directory="./renewal_dataset_0326_myeonmok/offgrounds",
        target_size=(512, 512),
        batch_size=1,
        shuffle=False,
        color_mode="grayscale",
        class_mode=None
    )


    def create_pred_generator(img, offground, ground):
        while True:
            for x1, x2, x3 in zip(img, offground, ground):
                yield (x1, x2, x3)


    pred_gen = create_pred_generator(pred_img_generator, pred_offground_generator, pred_ground_generator)

    DA_DRU = DA_DeepResUNet(input_shape=INPUT_SHAPE, lr=LR, num_classes=NUM_CLASSES, offground_shape=OFFGROUND_SHAPE, ground_shape=GROUND_SHAPE)
    model = DA_DRU.build_net()
    model.load_weights("./DA_DRU_rgbd_weights/DRU_97.hdf5")
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
    #     tf.keras.callbacks.ModelCheckpoint(filepath="./DA_DRU_rgbd_weights/DRU_{epoch:02d}.hdf5"),
    #     tf.keras.callbacks.TensorBoard(log_dir="./DA_DRU_rgbd_logs", update_freq="batch"),
    #     # VISUALIZE
    #     # keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
    # ]
    #
    # model.fit(
    #     train_gen,
    #     validation_data=valid_gen,
    #     validation_batch_size=BATCH_SIZE,
    #     validation_steps=valid_img_generator.samples / BATCH_SIZE,
    #     steps_per_epoch=mask_generator.samples / BATCH_SIZE,
    #     epochs=100,
    #     callbacks=[callbacks]
    # )

    for data, name in zip(pred_gen, pred_img_generator.filenames) :
        name = str(name).split('/')[-1].split('\\')[-1].split('.')[0]
        pred_test = model.predict(
           data,
           batch_size=1,
           verbose=1,
           steps=pred_img_generator.samples/1.
        )
        print(pred_test.shape)
        class2color(pred_test, name)
