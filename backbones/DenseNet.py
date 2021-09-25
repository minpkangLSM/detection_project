import numpy as np
from utils import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D, GlobalAvgPool2D
from tensorflow.keras.layers import Concatenate, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.applications import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.utils import losses_utils

# def hierarchical_regularization()

def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1):

  y_pred = ops.convert_to_tensor_v2(y_pred)
  print(y_pred.shape)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  return K.sparse_categorical_crossentropy(
      y_true, y_pred, from_logits=from_logits, axis=axis)

class UpdatedSparseCategoricalCrossentropy(SparseCategoricalCrossentropy):

    def __init__(self,
                 from_logits=False,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='sparse_categorical_crossentropy'):
        super(SparseCategoricalCrossentropy, self).__init__(
            sparse_categorical_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits)

class DenseNet121_imagenet :

    def __init__(self, input_shape, num_classes, k, lr):
        """
        DenseNet121 model.
        :param input_shape: input_shape, dtype = tuple
        :param num_classes: the number of class
        :param k: growth rate of the network
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.k = k
        self.lr = lr

    def build_net(self):

        input = keras.Input(shape=self.input_shape)

        # Head
        conv_init = Conv2D(filters=2*self.k, kernel_size=(7,7), kernel_initializer="he_normal",
                           strides=(2,2), padding="same", name="conv_init")(input)
        batch_init = BatchNormalization()(conv_init)
        activ_init = Activation(activation="relu")(batch_init)
        pool_init = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same", name="pool_init")(activ_init)

        # body
        dense_block_1 = Denseblock(input=pool_init, L=6, k=self.k, block_name="block1")
        trans_layer_1 = transition_layer(input=dense_block_1, layer_name="trans1")

        dense_block_2 = Denseblock(input=trans_layer_1, L=12, k=self.k, block_name="block2")
        trans_layer_2 = transition_layer(input=dense_block_2, layer_name="trans2")

        dense_block_3 = Denseblock(input=trans_layer_2, L=24, k=self.k, block_name="block3")
        trans_layer_3 = transition_layer(input=dense_block_3, layer_name="trans3")

        dense_block_4 = Denseblock(input=trans_layer_3, L=16, k=self.k, block_name="block4")

        # tail
        GAP = GlobalAvgPool2D()(dense_block_4)
        tail_dense = Dense(units=self.num_classes, activation="softmax")(GAP)

        model = keras.Model(inputs=input, outputs=tail_dense)
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=SparseCategoricalCrossentropy(),
            metrics=SparseCategoricalAccuracy()
        )

        return model

class DenseNet201_imagenet :

    def __init__(self, input_shape, num_classes, k, lr):
        """
        DenseNet121 model.
        :param input_shape: input_shape, dtype = tuple
        :param num_classes: the number of class
        :param k: growth rate of the network
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.k = k
        self.lr = lr

    def build_net(self):

        input = keras.Input(shape=self.input_shape)

        # head
        conv_init = Conv2D(filters=2*self.k, kernel_size=(7,7), kernel_initializer="he_normal",
                           strides=2, padding="same", name="conv_init")(input)
        batch_init = BatchNormalization()(conv_init)
        activ_init = Activation(activation="relu")(batch_init)
        pool_init = MaxPool2D(pool_size=(3,3), strides=(2,2), padding="same", name="pool_init")(activ_init)

        # body
        dense_block_1 = Denseblock(input=pool_init, L=6, k=self.k, block_name="dense_block_1")
        trans_layer_1 = transition_layer(input=dense_block_1, layer_name="trans1")

        dense_block_2 = Denseblock(input=trans_layer_1, L=12, k=self.k, block_name="dense_block_2")
        trans_layer_2 = transition_layer(input=dense_block_2, layer_name="trans2")

        dense_block_3 = Denseblock(input=trans_layer_2, L=48, k=self.k, block_name="dense_block_3")
        trans_layer_3 = transition_layer(input=dense_block_3, layer_name="trans3")

        dense_block_4 = Denseblock(input=trans_layer_3, L=32, k=self.k, block_name="dense_block_4")

        # tail - classifier
        GAP = GlobalAvgPool2D()(dense_block_4)
        tail_dense = Dense(units=self.num_classes, activation="softmax")(GAP)

        model = keras.Model(inputs=input, outputs=tail_dense)
        model.compile(
            optimizer = keras.optimizers.Adam(self.lr),
            loss = SparseCategoricalCrossentropy(),
            metrics=SparseCategoricalAccuracy()
        )

        return model

class DenseNet_cifar_L100k12_BC :

    def __init__(self, input_shape, num_classes, k, lr):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lr = lr
        self.k = k

    def build_net(self):

        input = keras.Input(shape=self.input_shape)

        # head
        conv_init = Conv2D(filters=self.k*2, kernel_size=(7,7), kernel_initializer="he_normal",
                           strides=(1,1), padding="same", name="conv_init")(input)
        batch_init = BatchNormalization()(conv_init)
        activ_init = Activation(activation="relu")(batch_init)
        pool_init = MaxPool2D(pool_size=(3,3), strides=(1,1), padding="same", name="pool_init")(activ_init)

        # body
        dense_block_1 = Denseblock(input=pool_init, L=40, k=self.k, block_name="dense_block_1")
        trans_layer_1 = transition_layer(input=dense_block_1, layer_name="trans_1")

        dense_block_2 = Denseblock(input=trans_layer_1, L=40, k=self.k, block_name="dense_block_2")
        trans_layer_2 = transition_layer(input=dense_block_2, layer_name="trans_2")

        dense_block_3 = Denseblock(input=trans_layer_2, L=40, k=self.k, block_name="dense_block_3")
        trans_layer_3 = transition_layer(input=dense_block_3, layer_name="trans_3")

        GAP = GlobalAvgPool2D()(trans_layer_3)
        tail_dense = Dense(units=self.num_classes, activation="softmax")(GAP)

        model = keras.Model(inputs=input, outputs=tail_dense)
        model.compile(
            optimizer = keras.optimizers.Adam(self.lr),
            loss = UpdatedSparseCategoricalCrossentropy(),
            metrics = SparseCategoricalAccuracy()
        )

        return model


if __name__ == "__main__" :

    # model
    EPOCH_ITER = 30
    BATCH_SIZE = 64
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    datagen = ImageDataGenerator()
    datagen.fit(x_train)

    instance = DenseNet_cifar_L100k12_BC(input_shape=(32,32,3), num_classes=10, k=12, lr=0.01)
    model = instance.build_net()
    model.summary()
    callbacks = [keras.callbacks.TensorBoard(log_dir="./logs/densenet", update_freq="batch")]
    model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(x_train) / BATCH_SIZE,
        epochs=EPOCH_ITER,
        validation_data=(x_val, y_val),
        callbacks=callbacks
    )