"""
2021.05.21. ~
Implementation of "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection", Hao Chen etal., 2019.
by Kangmin Park, Lab. for Sensor and Modeling, Geoinformatics, Univ. of Seoul.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import AvgPool2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import MeanIoU
from tensorflow.python.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.applications import ResNet50

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
        # y_pred = tf.math.argmax(y_pred, axis=-1)
        mask = y_pred > 1
        y_pred = tf.where(mask, y_pred, 1)
        return super().update_state(y_true, y_pred, sample_weight)

class STANetMeanIoU(MeanIoU) :
    def __init__(self,
                 y_true=None,
                 y_pred=None,
                 num_classes=None,
                 name=None,
                 dtype=None):
        super(STANetMeanIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.greater(y_pred, tf.ones_like(y_pred))
        y_pred = tf.cast(y_pred, tf.int8)
        return super().update_state(y_true, y_pred, sample_weight)

def scheduler(epoch,
              lr):

    initial_lr = 0.0001
    end_lr = 0.000001
    decay_step = 100
    lr = (initial_lr-end_lr)*(1-epoch/decay_step)+end_lr
    return lr

class MyCallBack(Callback):

    def on_epoch_end(self,
                     epoch,
                     logs=None):
        print(self.model.optimizer.lr)

def Resblock(input,
             knum,
             layer_name,
             pad="same",
             verbose=False):

    #identity mapping
    identity = input
    if verbose :
        identity = MaxPool2D(pool_size=1, strides=2)(identity)
        zero_pad = K.zeros_like(identity)
        identity = Concatenate()([identity, zero_pad])

    if not verbose :
        Conv_L1 = Conv2D(filters=knum, kernel_size=3, kernel_initializer="he_normal",
                         strides=1, padding=pad, name=layer_name+"_C_L1")(input)
    else :
        Conv_L1 = Conv2D(filters=knum, kernel_size=3, kernel_initializer="he_normal",
                         strides=2, padding=pad, name=layer_name+"_C_L1")(input)
    BN_L1 = BatchNormalization()(Conv_L1)
    AC_L1 = Activation(activation="relu")(BN_L1)

    Conv_L2 = Conv2D(filters=knum, kernel_size=3, kernel_initializer="he_normal",
                     strides=1, padding=pad, name=layer_name+"_C_L2")(AC_L1)
    BN_L2 = BatchNormalization()(Conv_L2)

    #shortcut
    shortcut = Add()([BN_L2, identity])
    shortcut = Activation(activation="relu")(shortcut)

    return shortcut

class feature_extractor() :

    def __init__(self):

        return

    def resnet18_stanet(self,
                        input_shape,
                        c1=96,
                        c2=256,
                        c3=64):

        input = keras.Input(shape=input_shape)
        # conv1 output = 1/2
        conv1 = Conv2D(filters=64,
                       kernel_size=7,
                       strides=2,
                       activation="gelu",
                       padding="same",
                       name="conv1")(input)

        # conv1_pool output = 1/4
        conv1_pool = MaxPool2D(pool_size=3,
                               strides=2,
                               padding="same",
                               name="conv1_pool1")(conv1)
        # res_block1 input / output size : 1/4 -> 1/4
        res_block1 = Resblock(input=conv1_pool,
                              knum=64,
                              layer_name="res_block1")
        # res_block2 input / output size 1/4 -> 1/8
        res_block2 = Resblock(input=res_block1,
                              knum=128,
                              layer_name="res_block2",
                              verbose=True)
        # res_block3 = input / output size 1/8 -> 1/16
        res_block3 = Resblock(input=res_block2,
                              knum=256,
                              layer_name="res_block3",
                              verbose=True)
        # res_block4 = input / output size 1/16 -> 1/32
        res_block4 = Resblock(input=res_block3,
                              knum=512,
                              layer_name="res_block4",
                              verbose=True)
        # res_block1
        res_block1_c = Conv2D(filters=c1,
                              kernel_size=1,
                              strides=1,
                              padding="valid")(res_block1)
        res_block1_c = BatchNormalization()(res_block1_c)
        res_block1_c = Activation(activation="gelu")(res_block1_c)

        # res_block2
        res_block2_c = Conv2D(filters=c1,
                              kernel_size=1,
                              strides=1,
                              padding="valid")(res_block2)
        res_block2_c = BatchNormalization()(res_block2_c)
        res_block2_c = Activation(activation="gelu")(res_block2_c)
        res_block2_c = Conv2DTranspose(filters=c1,
                                       kernel_size=3,
                                       padding="same",
                                       strides=2)(res_block2_c)
        # res_block3
        res_block3_c = Conv2D(kernel_size=1,
                              filters=c1,
                              strides=1,
                              padding="valid")(res_block3)
        res_block3_c = BatchNormalization()(res_block3_c)
        res_block3_c = Activation(activation="gelu")(res_block3_c)
        res_block3_c = Conv2DTranspose(filters=c1,
                                       kernel_size=3,
                                       padding="same",
                                       strides=4)(res_block3_c)
        # res_block1
        res_block4_c = Conv2D(kernel_size=1,
                              filters=c1,
                              strides=1,
                              padding="valid")(res_block4)
        res_block4_c = BatchNormalization()(res_block4_c)
        res_block4_c = Activation(activation="gelu")(res_block4_c)
        res_block4_c = Conv2DTranspose(filters=c1,
                                       kernel_size=3,
                                       padding="same",
                                       strides=8)(res_block4_c)

        concat = Concatenate()([res_block1_c,
                                res_block2_c,
                                res_block3_c,
                                res_block4_c])

        last_conv3 = Conv2D(filters=c2,
                            kernel_size=3,
                            strides=1,
                            padding="same")(concat)
        last_conv1 = Conv2D(filters=c3,
                            kernel_size=1,
                            strides=1,
                            padding="same")(last_conv3)

        model = keras.Model(inputs=input,
                            outputs=last_conv1)

        return model

    def resnet50_stanet(self,
                        input_shape,
                        imagenet_trainable,
                        c1=96,
                        c2=256,
                        c3=64):
        input = keras.Input(input_shape)
        resnet50 = ResNet50(input_tensor=input,
                            include_top=False,
                            weights="imagenet")
        resnet50.trainable=imagenet_trainable

        RB16 = resnet50.get_layer(name="conv5_block3_out").output
        RB16 = Conv2D(filters=c1,
                      kernel_size=1,
                      strides=1,
                      padding="valid")(RB16)
        RB16 = BatchNormalization()(RB16)
        RB16 = Activation(activation="relu")(RB16)

        RB13 = resnet50.get_layer(name="conv4_block6_out").output
        RB13 = Conv2D(filters=c1,
                      kernel_size=1,
                      strides=1,
                      padding="valid")(RB13)
        RB13 = BatchNormalization()(RB13)
        RB13 = Activation(activation="relu")(RB13)

        RB7 = resnet50.get_layer(name="conv3_block4_out").output
        RB7 = Conv2D(filters=c1,
                      kernel_size=1,
                      strides=1,
                      padding="valid")(RB7)
        RB7 = BatchNormalization()(RB7)
        RB7 = Activation(activation="relu")(RB7)

        RB3 = resnet50.get_layer(name="conv2_block3_out").output
        RB3 = Conv2D(filters=c1,
                      kernel_size=1,
                      strides=1,
                      padding="valid")(RB3)
        RB3 = BatchNormalization()(RB3)
        RB3 = Activation(activation="relu")(RB3)

        RB16 = UpSampling2D(size=(8, 8),
                            interpolation="bilinear")(RB16)
        RB13 = UpSampling2D(size=(4, 4),
                            interpolation="bilinear")(RB13)
        RB7 = UpSampling2D(size=(2, 2),
                           interpolation="bilinear")(RB7)

        feature = Concatenate(axis=-1)([RB16, RB13, RB7, RB3])
        feature = Conv2D(filters=c2,
                         kernel_size=3,
                         strides=1,
                         padding="same")(feature)
        feature = BatchNormalization()(feature)
        feature = Activation(activation="relu")(feature)
        feature = Conv2D(filters=c3,
                         kernel_size=3,
                         strides=1,
                         padding="same")(feature)
        feature = BatchNormalization()(feature)
        feature = Activation(activation="relu")(feature)

        model = keras.Model(inputs=input,
                            outputs=feature)

        return model

def bam(input_tensor,
        name="bam_") :

    # 두 시점의 피쳐가 동시에 conv 연산이 돼서는 안된다.
    # input_tensor : [batch, height, width*2, channel]
    # K, Q, V
    x = input_tensor
    h, w, c = x.shape[1], x.shape[2], x.shape[3] # h=h, w=2w, c=c

    Q = Conv2D(filters=c//8,
               kernel_size=1,
               strides=1,
               padding="valid")(x)
    Q = Activation(activation="gelu")(Q)
    Q = BatchNormalization()(Q)
    Q = tf.reshape(Q,
                   shape=[-1, h*w, c//8])

    K = Conv2D(filters=c//8,
               kernel_size=1,
               strides=1,
               padding="valid")(x)
    K = Activation(activation="gelu")(K)
    K = BatchNormalization()(K)
    K = tf.reshape(K,
                   shape=[-1, h*w, c//8])

    V = Conv2D(filters=c,
               kernel_size=1,
               strides=1,
               padding="valid")(x)
    V = Activation(activation="gelu")(V)
    V = BatchNormalization()(V)
    V = tf.reshape(V,
                   shape=[-1, h*w, c])

    attention = tf.matmul(Q, tf.transpose(K, perm=[0, 2, 1]))/tf.math.sqrt(tf.cast(c//8, tf.float32))
    attention = tf.keras.activations.softmax(attention, axis=-1)

    attention_score = tf.matmul(attention, V)
    attention_score = tf.reshape(attention_score,
                                 shape=[-1, h, w, c])
    bam = attention_score + x

    return bam

def pam(input_tensor,
        k_channels,
        v_channels,
        scales = (1,2,4,8),
        d_sample = 1):

    context_list = []

    for scale in scales :

        # input shape = [b, h, w*2, c]
        if d_sample != 1 : input_tensor = AvgPool2D(pool_size=(d_sample, d_sample))(input_tensor)
        b, h, w, c = input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2] // 2, input_tensor.shape[3]

        local_y = []
        local_x = []
        step_h, step_w = h // scale, w // scale
        for i in range(0, scale):
            for j in range(0, scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, h), min(start_y + step_w, w)
                if i == (scale - 1):
                    end_x = h
                if j == (scale - 1):
                    end_y = w
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]

        # Key, Query, Value
        K = Conv2D(filters=k_channels,
                   kernel_size=1,
                   strides=1,
                   padding="valid")(input_tensor)
        K = Activation(activation="gelu")(K)
        K = BatchNormalization()(K)

        Q = Conv2D(filters=k_channels,
                   kernel_size=1,
                   strides=1,
                   padding="valid")(input_tensor)
        Q = Activation(activation="gelu")(Q)
        Q = BatchNormalization()(Q)

        V = Conv2D(filters=v_channels,
                   kernel_size=1,
                   strides=1,
                   padding="valid")(input_tensor)
        V = Activation(activation="gelu")(V)
        V = BatchNormalization()(V)

        K = tf.stack([K[:, :, :w, :], K[:, :, w:, :]], axis=4)
        Q = tf.stack([Q[:, :, :w, :], Q[:, :, w:, :]], axis=4)
        V = tf.stack([V[:, :, :w, :], V[:, :, w:, :]], axis=4)

        def func(value_local, query_local, key_local):

            h_local, w_local = value_local.shape[1], value_local.shape[2]

            value_local = tf.reshape(value_local,
                                     shape=[-1, v_channels, h_local*w_local*2])
            query_local = tf.reshape(query_local,
                                     shape=[-1, k_channels, h_local*w_local*2])
            query_local = tf.transpose(query_local, perm=[0,2,1])
            key_local = tf.reshape(key_local,
                                   shape=[-1, k_channels, h_local*w_local*2])

            sim_map = tf.matmul(query_local, key_local)/tf.math.sqrt(tf.cast(k_channels, tf.float32))
            sim_map = tf.keras.activations.softmax(sim_map, axis=-1)
            context_local = tf.matmul(value_local, tf.transpose(sim_map, perm=[0, 2, 1]))
            context_local = tf.reshape(context_local,
                                       shape=[-1, h_local, w_local, v_channels, 2])
            return context_local

        local_block_cnt = scale*scale*2

        v_list = [V[:, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1], :] for i in
                  range(0, local_block_cnt, 2)]
        v_locals = tf.concat(v_list, axis=0)

        q_list = [Q[:, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1], :] for i in
                  range(0, local_block_cnt, 2)]
        q_locals = tf.concat(q_list, axis=0)

        k_list = [K[:, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1], :] for i in
                  range(0, local_block_cnt, 2)]
        k_locals = tf.concat(k_list, axis=0)

        context_locals = func(v_locals, q_locals, k_locals)
        context = tf.reshape(context_locals,
                             shape=[-1, h, 2*w, context_locals.shape[3]])
        if d_sample != 1 :
            context = UpSampling2D(size=(d_sample, d_sample),
                                   interpolation="bilinear")(context)

        #context = context+input_tensor
        context_list.append(context)

    context = Concatenate(axis=-1)(context_list)
    context = Conv2D(filters=c,#context.shape[-1],
                     kernel_size=1,
                     strides=1,
                     padding="valid")(context)
    context = Activation(activation="gelu")(context)
    context = BatchNormalization()(context)
    context = context + input_tensor
    return context

def contrastive_loss_reverse(y_true, y_pred):
    """
    batch-balanced contrastive loss(BCL) version
    :param y_true:
    :param y_pred:
    :return:
    """
    # set margin
    margin = 2

    # the number of pixels in ground truth for each class (binary)
    num_of_change = tf.reduce_sum(y_true)+K.epsilon()
    num_of_nochange = tf.reduce_sum(1-y_true)+K.epsilon()

    # change
    change = tf.reduce_sum(y_true*tf.pow(tf.maximum(0., margin - y_pred),2))/num_of_change
    # no change parts
    no_change = tf.reduce_sum((1-y_true)*tf.pow(y_pred, 2))/num_of_nochange

    loss = change + no_change
    return loss

class STANet :

    def __init__(self,
                 input_shape,
                 learning_rate,
                 num_classes):
        self.input_shape = input_shape
        self.lr = learning_rate
        self.num_classes = num_classes
        return

    def build_net(self):

        feature_extract = feature_extractor()
        resnet18 = feature_extract.resnet18_stanet(input_shape=self.input_shape)
        # resnet50 = feature_extract.resnet50_stanet(input_shape=self.input_shape,
        #                                            imagenet_trainable=True)

        input1 = keras.Input(shape=self.input_shape)
        input2 = keras.Input(shape=self.input_shape)

        f1 = resnet18(input1)
        f2 = resnet18(input2)

        # [B, H, W, C]
        x = Concatenate(axis=2)([f1, f2])

        pam_z = pam(x,
                    k_channels=64//8,
                    v_channels=64)
        p_z1 = pam_z[:, :, :int(x.shape[2]//2), :]
        p_z2 = pam_z[:, :, int(x.shape[2]//2):, :]

        z1 = UpSampling2D(size=(4,4),
                          interpolation="bilinear")(p_z1)
        z2 = UpSampling2D(size=(4,4),
                          interpolation="bilinear")(p_z2)

        # channel of z1, z2 : 128
        dist_map = tf.math.reduce_euclidean_norm(z1 - z2, axis=-1, keepdims=True)

        model = keras.Model(inputs=(input1, input2),
                            outputs=dist_map)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                      loss=contrastive_loss_reverse,
                      metrics=STANetMeanIoU(num_classes=2))

        return model

if __name__ == "__main__" :

    INPUT_SHAPE = (256, 256, 3)
    NUM_CLASSES = 2
    LR = 0.0001
    BATCH_SIZE = 2
    seed = 10

    # train
    image_datagen_2019 = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
    )
    image_datagen_2018 = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
    )
    mask_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
    )

    image_generator_2019 = image_datagen_2019.flow_from_directory(
       directory="./levir/train/images/A",
       class_mode=None,
       seed=seed,
       shuffle=True,
       target_size=(256, 256),
       batch_size=BATCH_SIZE
    )
    image_generator_2018 = image_datagen_2018.flow_from_directory(
        directory="./levir/train/images/B",
        class_mode=None,
        seed=seed,
        shuffle=True,
        target_size=(256, 256),
        batch_size=BATCH_SIZE
    )
    mask_generator = mask_datagen.flow_from_directory(
       directory="./levir/train/labels",
       class_mode=None,
       seed=seed,
       shuffle=True,
       target_size=(256, 256),
       color_mode="grayscale",
       batch_size=BATCH_SIZE
    )

    # valid
    image_datagen_2019 = ImageDataGenerator()
    image_datagen_2018 = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()

    valid_image_generator_2019 = image_datagen_2019.flow_from_directory(
        directory="./levir/valid/image/A",
        class_mode=None,
        seed=seed,
        shuffle=True,
        target_size=(256, 256),
        batch_size=BATCH_SIZE
    )
    valid_image_generator_2018 = image_datagen_2018.flow_from_directory(
        directory="./levir/valid/image/B",
        class_mode=None,
        seed=seed,
        shuffle=True,
        target_size=(256, 256),
        batch_size=BATCH_SIZE
    )
    valid_mask_generator = mask_datagen.flow_from_directory(
        directory="./levir/valid/label",
        class_mode=None,
        seed=seed,
        shuffle=True,
        target_size=(256, 256),
        color_mode="grayscale",
        batch_size=BATCH_SIZE
    )

    def create_pred_generator(img1, img2, label):
        while True:
            for x1, x2, x3 in zip(img1, img2, label):
                yield (x1, x2), x3

    stanet = STANet(input_shape=INPUT_SHAPE,
                    learning_rate=LR,
                    num_classes=NUM_CLASSES)
    model = stanet.build_net()
    model.summary()

    train_gen = create_pred_generator(image_generator_2018, image_generator_2019, mask_generator)
    valid_gen = create_pred_generator(valid_image_generator_2018, valid_image_generator_2019, valid_mask_generator)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath="./STANet_callback/weights/STANet_{epoch:02d}.hdf5"),
        tf.keras.callbacks.TensorBoard(log_dir="./STANet_callback/logs", update_freq="batch"),
        tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
        MyCallBack()
    ]

    model.fit(
        train_gen,
        batch_size=BATCH_SIZE,
        validation_data=valid_gen,
        validation_batch_size=BATCH_SIZE,
        validation_steps=valid_image_generator_2019.samples/BATCH_SIZE,
        steps_per_epoch=mask_generator.samples*4//BATCH_SIZE,
        epochs=100,
        callbacks=[callbacks]
    )