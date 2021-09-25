import numpy as np
from utils import *
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, AvgPool2D, MaxPool2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, SparseCategoricalAccuracy
from keras.preprocessing.image import ImageDataGenerator
# reupdated
class ResNet34 :

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_net(self):

        inputs = keras.Input(shape=self.input_shape)

        # HEAD
        Conv = Conv2D(filters=64, kernel_size=7, strides=2, activation="relu", padding="same", name="Conv")(inputs)
        Pool = MaxPool2D(pool_size=3, strides=2, padding="same")(Conv)

        # BODY
        RB1 = Resblock(Pool, knum=64, layer_name="RB1")
        RB2 = Resblock(RB1, knum=64, layer_name="RB2")
        RB3 = Resblock(RB2, knum=64, layer_name="RB3")

        RB4 = Resblock(RB3, knum=128, layer_name="RB4", verbose=True)
        RB5 = Resblock(RB4, knum=128, layer_name="RB5")
        RB6 = Resblock(RB5, knum=128, layer_name="RB6")
        RB7 = Resblock(RB6, knum=128, layer_name="RB7")

        RB8 = Resblock(RB7, knum=256, layer_name="RB8", verbose=True)
        RB9 = Resblock(RB8, knum=256, layer_name="RB9")
        RB10 = Resblock(RB9, knum=256, layer_name="RB10")
        RB11 = Resblock(RB10, knum=256, layer_name="RB11")
        RB12 = Resblock(RB11, knum=256, layer_name="RB12")
        RB13 = Resblock(RB12, knum=256, layer_name="RB13")

        RB14 = Resblock(RB13, knum=512, layer_name="RB14", verbose=True)
        RB15 = Resblock(RB14, knum=512, layer_name="RB15")
        RB16 = Resblock(RB15, knum=512, layer_name="RB16")

        #TAIL
        TAIL_AVG = AvgPool2D(pool_size=3, strides=1, padding="same")(RB16)
        TAIL_FLT = Flatten()(TAIL_AVG)
        outputs = Dense(units=self.num_classes, activation="softmax")(TAIL_FLT)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer = keras.optimizers.Adam(0.002),
            loss = SparseCategoricalCrossentropy(),
            metrics = SparseCategoricalAccuracy()
        )

        return model

class ResNet50:
    """
    based on Bottleneck Ver. Residual block.
    """
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_net(self):

        inputs = keras.Input(shape=self.input_shape)

        # FRONT
        Conv = Conv2D(filters=64, kernel_size=7, strides=2, padding="same", activation="relu", name="Conv")(inputs)
        Pool = MaxPool2D(pool_size=3, strides=2, padding="same")(Conv)

        # BODY
        RB1 = Resblock_bn(input=Pool, knum_in=64, knum_out=256, layer_name="RB1")
        RB2 = Resblock_bn(input=RB1, knum_in=64, knum_out=256, layer_name="RB2")
        RB3 = Resblock_bn(input=RB2, knum_in=64, knum_out=256, layer_name="RB3")

        RB4 = Resblock_bn(input=RB3, knum_in=128, knum_out=512, layer_name="RB4", verbose=True)
        RB5 = Resblock_bn(input=RB4, knum_in=128, knum_out=512, layer_name="RB5")
        RB6 = Resblock_bn(input=RB5, knum_in=128, knum_out=512, layer_name="RB6")
        RB7 = Resblock_bn(input=RB6, knum_in=128, knum_out=512, layer_name="RB7")

        RB8 = Resblock_bn(input=RB7, knum_in=256, knum_out=1024, layer_name="RB8", verbose=True)
        RB9 = Resblock_bn(input=RB8, knum_in=256, knum_out=1024, layer_name="RB9")
        RB10 = Resblock_bn(input=RB9, knum_in=256, knum_out=1024, layer_name="RB10")
        RB11 = Resblock_bn(input=RB10,knum_in=256, knum_out=1024, layer_name="RB11")
        RB12 = Resblock_bn(input=RB11, knum_in=256, knum_out=1024, layer_name="RB12")
        RB13 = Resblock_bn(input=RB12, knum_in=256, knum_out=1024, layer_name="RB13")

        RB14 = Resblock_bn(input=RB13, knum_in=512, knum_out=2048, layer_name="RB14", verbose=True)
        RB15 = Resblock_bn(input=RB14, knum_in=512, knum_out=2048, layer_name="RB15")
        RB16 = Resblock_bn(input=RB15, knum_in=512, knum_out=2048, layer_name="RB16")

        # TAIL
        TAIL_AVG = AvgPool2D(pool_size=3, strides=1, padding="same")(RB16)
        TAIL_FLT = Flatten()(TAIL_AVG)
        outputs = Dense(units=self.num_classes, activation="softmax")(TAIL_FLT)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(0.002),
            loss = SparseCategoricalCrossentropy(),
            metrics=SparseCategoricalAccuracy()
        )

        return model

class ResNet101:

    def __init__(self, image_shape, num_classes):
        self.image_shape = image_shape
        self.num_classes = num_classes

    def build_net(self):
        input = keras.Input(self.image_shape)

        # HEAD
        Conv = Conv2D(filters=64, kernel_size=7, kernel_initializer="relu",
                      strides=2, padding="same", name="HEAD_conv")(input)
        MaxP = MaxPool2D(pool_size=3, strides=2, padding="same")(Conv)

        # BODY
        # layers : 3
        RB1 = Resblock(input=MaxP, knum=64, layer_name="RB1")
        RB2 = Resblock(input=RB1, knum=64, layer_name="RB2")
        RB3 = Resblock(input=RB2, knum=64, layer_name="RB3")
        # layers : 4
        RB4 = Resblock(input=RB3, knum=128, layer_name="RB4")
        RB5 = Resblock(input=RB4, knum=128, layer_name="RB5")
        RB6 = Resblock(input=RB5, knum=128, layer_name="RB6")
        RB7 = Resblock(input=RB6, knum=128, layer_name="RB7")
        # layers : 23
        RB8 = Resblock(input=RB7, knum=256, layer_name="RB8")
        RB9 = Resblock(input=RB8, knum=256, layer_name="RB9")
        RB10 = Resblock(input=RB9, knum=256, layer_name="RB10")
        RB11 = Resblock(input=RB10, knum=256, layer_name="RB11")
        RB12 = Resblock(input=RB11, knum=256, layer_name="RB12")
        RB13 = Resblock(input=RB12, knum=256, layer_name="RB13")
        RB14 = Resblock(input=RB13, knum=256, layer_name="RB14")
        RB15 = Resblock(input=RB14, knum=256, layer_name="RB15")
        RB16 = Resblock(input=RB15, knum=256, layer_name="RB16")
        RB17 = Resblock(input=RB16, knum=256, layer_name="RB17")
        RB18 = Resblock(input=RB17, knum=256, layer_name="RB18")
        RB19 = Resblock(input=RB18, knum=256, layer_name="RB19")
        RB20 = Resblock(input=RB19, knum=256, layer_name="RB20")
        RB21 = Resblock(input=RB20, knum=256, layer_name="RB21")
        RB22 = Resblock(input=RB21, knum=256, layer_name="RB22")
        RB23 = Resblock(input=RB22, knum=256, layer_name="RB23")
        RB24 = Resblock(input=RB23, knum=256, layer_name="RB24")
        RB25 = Resblock(input=RB24, knum=256, layer_name="RB25")
        RB26 = Resblock(input=RB25, knum=256, layer_name="RB26")
        RB27 = Resblock(input=RB26, knum=256, layer_name="RB27")
        RB28 = Resblock(input=RB27, knum=256, layer_name="RB28")
        RB29 = Resblock(input=RB28, knum=256, layer_name="RB29")
        RB30 = Resblock(input=RB29, knum=256, layer_name="RB30")
        # layers : 3
        RB31 = Resblock(input=RB30, knum=512, layer_name="RB31")
        RB32 = Resblock(input=RB31, knum=512, layer_name="RB32")
        RB33 = Resblock(input=RB32, knum=512, layer_name="RB33")

        # TAIL
        TAIL_AVG = AvgPool2D(pool_size=3, strides=1, padding="same")(RB33)
        TAIL_FLT = Flatten()(TAIL_AVG)
        outputs = Dense(units=self.num_classes, activation="softmax")(TAIL_FLT)

        model = keras.Model(inputs=input, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(0.002),
            loss=SparseCategoricalCrossentropy(),
            metrics=SparseCategoricalAccuracy()
        )

        return model

class Resnet20_cifar(ResNet34):

    def build_net(self):

        inputs = keras.Input(shape=self.input_shape)

        #HEAD(NO-Pool)
        Conv = Conv2D(filters=16, kernel_size=3, kernel_initializer="he_normal",
                      strides=1, activation="relu", padding="same", name="Conv")(inputs)

        #BODY
        RB1 = Resblock(input=Conv, knum=16, layer_name="RB1")
        RB2 = Resblock(input=RB1, knum=16, layer_name="RB2")
        RB3 = Resblock(input=RB2, knum=16, layer_name="RB3")
        RB4 = Resblock(input=RB3, knum=16, layer_name="RB4")
        RB5 = Resblock(input=RB4, knum=16, layer_name="RB5")
        RB6 = Resblock(input=RB5, knum=16, layer_name="RB6")

        RB7 = Resblock(input=RB6, knum=32, layer_name="RB7", verbose=True)
        RB8 = Resblock(input=RB7, knum=32, layer_name="RB8")
        RB9 = Resblock(input=RB8, knum=32, layer_name="RB9")
        RB10 = Resblock(input=RB9, knum=32, layer_name="RB10")
        RB11 = Resblock(input=RB10, knum=32, layer_name="RB11")
        RB12 = Resblock(input=RB11, knum=32, layer_name="RB12")

        RB13 = Resblock(input=RB12, knum=64, layer_name="RB13", verbose=True)
        RB14 = Resblock(input=RB13, knum=64, layer_name="RB14")
        RB15 = Resblock(input=RB14, knum=64, layer_name="RB15")
        RB16 = Resblock(input=RB15, knum=64, layer_name="RB16")
        RB17 = Resblock(input=RB16, knum=64, layer_name="RB17")
        RB18 = Resblock(input=RB17, knum=64, layer_name="RB18")

        #TAIL
        TAIL_AVG = AvgPool2D(pool_size=3, strides=1, padding="same")(RB18)
        TAIL_FLT = Flatten()(TAIL_AVG)
        outputs = Dense(units=self.num_classes, activation="softmax")(TAIL_FLT)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=SparseCategoricalCrossentropy(),
            metrics=SparseCategoricalAccuracy()
        )

        return model

class Resnet56_cifar(ResNet34):

    def build_net(self):
        inputs = keras.Input(shape=self.input_shape)

        # HEAD(NO-Pool)
        Conv = Conv2D(filters=16, kernel_size=3, kernel_initializer="he_normal",
                      strides=1, activation="relu", padding="same", name="Conv")(inputs)

        # BODY
        RB1 = Resblock(input=Conv, knum=16, layer_name="RB1")
        RB2 = Resblock(input=RB1, knum=16, layer_name="RB2")
        RB3 = Resblock(input=RB2, knum=16, layer_name="RB3")
        RB4 = Resblock(input=RB3, knum=16, layer_name="RB4")
        RB5 = Resblock(input=RB4, knum=16, layer_name="RB5")
        RB6 = Resblock(input=RB5, knum=16, layer_name="RB6")
        RB7 = Resblock(input=RB6, knum=16, layer_name="RB7")
        RB8 = Resblock(input=RB7, knum=16, layer_name="RB8")
        RB9 = Resblock(input=RB8, knum=16, layer_name="RB9")

        RB10 = Resblock(input=RB9, knum=32, layer_name="RB10", verbose=True)
        RB11 = Resblock(input=RB10, knum=32, layer_name="RB11")
        RB12 = Resblock(input=RB11, knum=32, layer_name="RB12")
        RB13 = Resblock(input=RB12, knum=32, layer_name="RB13")
        RB14 = Resblock(input=RB13, knum=32, layer_name="RB14")
        RB15 = Resblock(input=RB14, knum=32, layer_name="RB15")
        RB16 = Resblock(input=RB15, knum=32, layer_name="RB16")
        RB17 = Resblock(input=RB16, knum=32, layer_name="RB17")
        RB18 = Resblock(input=RB17, knum=32, layer_name="RB18")

        RB19 = Resblock(input=RB18, knum=64, layer_name="RB19", verbose=True)
        RB20 = Resblock(input=RB19, knum=64, layer_name="RB20")
        RB21 = Resblock(input=RB20, knum=64, layer_name="RB21")
        RB22 = Resblock(input=RB21, knum=64, layer_name="RB22")
        RB23 = Resblock(input=RB22, knum=64, layer_name="RB23")
        RB24 = Resblock(input=RB23, knum=64, layer_name="RB24")
        RB25 = Resblock(input=RB24, knum=64, layer_name="RB25")
        RB26 = Resblock(input=RB25, knum=64, layer_name="RB26")
        RB27 = Resblock(input=RB26, knum=64, layer_name="RB27")

        # TAIL
        TAIL_AVG = AvgPool2D(pool_size=3, strides=1, padding="same")(RB27)
        TAIL_FLT = Flatten()(TAIL_AVG)
        outputs = Dense(units=self.num_classes, activation="softmax")(TAIL_FLT)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=SparseCategoricalCrossentropy(),
            metrics=SparseCategoricalAccuracy()
        )

        return model

class Resnet110_cifar(ResNet34):

    def build_net(self):
        inputs = keras.Input(shape=self.input_shape)

        # HEAD(NO-Pool)
        Conv = Conv2D(filters=16, kernel_size=3, kernel_initializer="he_normal",
                      strides=1, activation="relu", padding="same", name="Conv")(inputs)

        # BODY
        RB1 = Resblock(input=Conv, knum=16, layer_name="RB1")
        RB2 = Resblock(input=RB1, knum=16, layer_name="RB2")
        RB3 = Resblock(input=RB2, knum=16, layer_name="RB3")
        RB4 = Resblock(input=RB3, knum=16, layer_name="RB4")
        RB5 = Resblock(input=RB4, knum=16, layer_name="RB5")
        RB6 = Resblock(input=RB5, knum=16, layer_name="RB6")
        RB7 = Resblock(input=RB6, knum=16, layer_name="RB7")
        RB8 = Resblock(input=RB7, knum=16, layer_name="RB8")
        RB9 = Resblock(input=RB8, knum=16, layer_name="RB9")
        RB10 = Resblock(input=RB9, knum=16, layer_name="RB10")
        RB11 = Resblock(input=RB10, knum=16, layer_name="RB11")
        RB12 = Resblock(input=RB11, knum=16, layer_name="RB12")
        RB13 = Resblock(input=RB12, knum=16, layer_name="RB13")
        RB14 = Resblock(input=RB13, knum=16, layer_name="RB14")
        RB15 = Resblock(input=RB14, knum=16, layer_name="RB15")
        RB16 = Resblock(input=RB15, knum=16, layer_name="RB16")
        RB17 = Resblock(input=RB16, knum=16, layer_name="RB17")
        RB18 = Resblock(input=RB17, knum=16, layer_name="RB18")

        RB19 = Resblock(input=RB18, knum=32, layer_name="RB19", verbose=True)
        RB20 = Resblock(input=RB19, knum=32, layer_name="RB20")
        RB21 = Resblock(input=RB20, knum=32, layer_name="RB21")
        RB22 = Resblock(input=RB21, knum=32, layer_name="RB22")
        RB23 = Resblock(input=RB22, knum=32, layer_name="RB23")
        RB24 = Resblock(input=RB23, knum=32, layer_name="RB24")
        RB25 = Resblock(input=RB24, knum=32, layer_name="RB25")
        RB26 = Resblock(input=RB25, knum=32, layer_name="RB26")
        RB27 = Resblock(input=RB26, knum=32, layer_name="RB27")
        RB28 = Resblock(input=RB27, knum=32, layer_name="RB28")
        RB29 = Resblock(input=RB28, knum=32, layer_name="RB29")
        RB30 = Resblock(input=RB29, knum=32, layer_name="RB30")
        RB31 = Resblock(input=RB30, knum=32, layer_name="RB31")
        RB32 = Resblock(input=RB31, knum=32, layer_name="RB32")
        RB33 = Resblock(input=RB32, knum=32, layer_name="RB33")
        RB34 = Resblock(input=RB33, knum=32, layer_name="RB34")
        RB35 = Resblock(input=RB34, knum=32, layer_name="RB35")
        RB36 = Resblock(input=RB35, knum=32, layer_name="RB36")

        RB37 = Resblock(input=RB36, knum=64, layer_name="RB37", verbose=True)
        RB38 = Resblock(input=RB37, knum=64, layer_name="RB38")
        RB39 = Resblock(input=RB38, knum=64, layer_name="RB39")
        RB40 = Resblock(input=RB39, knum=64, layer_name="RB40")
        RB41 = Resblock(input=RB40, knum=64, layer_name="RB41")
        RB42 = Resblock(input=RB41, knum=64, layer_name="RB42")
        RB43 = Resblock(input=RB42, knum=64, layer_name="RB43")
        RB44 = Resblock(input=RB43, knum=64, layer_name="RB44")
        RB45 = Resblock(input=RB44, knum=64, layer_name="RB45")
        RB46 = Resblock(input=RB45, knum=64, layer_name="RB46")
        RB47 = Resblock(input=RB46, knum=64, layer_name="RB47")
        RB48 = Resblock(input=RB47, knum=64, layer_name="RB48")
        RB49 = Resblock(input=RB48, knum=64, layer_name="RB49")
        RB50 = Resblock(input=RB49, knum=64, layer_name="RB50")
        RB51 = Resblock(input=RB50, knum=64, layer_name="RB51")
        RB52 = Resblock(input=RB51, knum=64, layer_name="RB52")
        RB53 = Resblock(input=RB52, knum=64, layer_name="RB53")
        RB54 = Resblock(input=RB53, knum=64, layer_name="RB54")

        # TAIL
        TAIL_AVG = AvgPool2D(pool_size=3, strides=1, padding="same")(RB54)
        TAIL_FLT = Flatten()(TAIL_AVG)
        outputs = Dense(units=self.num_classes, activation="softmax")(TAIL_FLT)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=SparseCategoricalCrossentropy(),
            metrics=SparseCategoricalAccuracy()
        )

        return model

if __name__ == "__main__" :

    EPOCH_ITER = 30
    BATCH_SIZE = 64
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    datagen = ImageDataGenerator()
    datagen.fit(x_train)

    Resnet = Resnet20_cifar(input_shape=(32,32,3), num_classes=10)
    model = Resnet.build_net()

    # Load Weights
    # model.load_weights(filepath="./resnet_weights/resnet20_02.hdf5", by_name=True)
    # model.trainable = False
    model.summary()
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath='./resnet_weights/resnet20_{epoch:02d}.hdf5'),
        keras.callbacks.TensorBoard(log_dir="./logs/resnet20_cifar",
                                    update_freq="batch")
    ]
    history = model.fit(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
              steps_per_epoch=len(x_train)/BATCH_SIZE,
              epochs=EPOCH_ITER,
              callbacks=[callbacks],
              validation_data=(x_val, y_val))

    # manual version
    # for epoch in range(EPOCH_ITER):
    #     print("EPOCH : ", epoch)
    #     batches = 0
    #     for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=64):
    #         model.fit(x_batch,
    #                   y_batch,
    #                   callbacks=[callbacks])
    #         batches+=1
    #         if batches >= len(x_train)/BATCH_SIZE : break