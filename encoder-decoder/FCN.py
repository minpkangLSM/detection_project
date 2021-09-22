import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Add
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.applications import VGG16

class FCN :

    def __init__(self, input_shape, num_classes, lr, mode):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lr = lr
        if mode in {"FCN32s", "FCN16s", "FCN8s"} :
            self.mode = mode
        else :
            raise ValueError("MODE has three types : FCN32s, FCN16s, FCN8s.")

    def build_net(self):

        input = keras.Input(self.input_shape)

        conv1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", name="conv1", activation="relu")(input)
        conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", name="conv2", activation="relu")(conv1)
        # output feature : /2 x /2 x 64
        pool1 = MaxPool2D(pool_size=(3,3), padding="same", strides=(2,2), name="pool1")(conv2)

        conv3 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", name="conv3", activation="relu")(pool1)
        conv4 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", name="conv4", activation="relu")(conv3)
        # output feature : /4 x /4 x 128
        pool2 = MaxPool2D(pool_size=(3,3), padding="same", strides=(2,2), name="pool2")(conv4)

        conv5 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", name="conv5", activation="relu")(pool2)
        conv6 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", name="conv6", activation="relu")(conv5)
        conv7 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", name="conv7", activation="relu")(conv6)
        # output feature : /8 x /8 x 256
        pool3 = MaxPool2D(pool_size=(3,3), padding="same", strides=(2,2), name="pool3")(conv7)

        conv8 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", name="conv8", activation="relu")(pool3)
        conv9 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", name="conv9", activation="relu")(conv8)
        conv10 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", name="conv10", activation="relu")(conv9)
        # output feature : /16 x /16 x 512
        pool4 = MaxPool2D(pool_size=(3, 3), padding="same", strides=(2, 2), name="pool4")(conv10)

        conv11 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", name="conv11", activation="relu")(pool4)
        conv12 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", name="conv12", activation="relu")(conv11)
        conv13 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", name="conv13", activation="relu")(conv12)
        # output feature : /32 x /32 x 512
        pool5 = MaxPool2D(pool_size=(3, 3), padding="same", strides=(2, 2), name="pool5")(conv13)

        # 1x1 convolution layers : FCN
        conv14 = Conv2D(filters=4096, kernel_size=(1, 1), strides=(1,1), padding="valid", name="conv14", activation="relu")(pool5)
        conv15 = Conv2D(filters=4096, kernel_size=(1, 1), strides=(1, 1), padding="valid", name="conv15", activation="relu")(conv14)
        conv16 = Conv2D(filters=self.num_classes, kernel_size=(1, 1), strides=(1, 1), padding="valid", name="conv16", activation="relu")(conv15)

        if self.mode == "FCN32s" :
            # output feature : 1/ x 1/ x 21
            output = UpSampling2D(size=(32, 32), interpolation="bilinear")(conv16)

        elif self.mode == "FCN16s" :
            upsample1 = UpSampling2D(size=(2, 2), interpolation="bilinear")(conv16)
            pool4 = Conv2D(filters=self.num_classes, kernel_size=(1,1), strides=(1,1),
                           padding="valid", name="pool4_mod")(pool4)
            sum = Add()([upsample1, pool4])
            output = UpSampling2D(size=(16, 16), interpolation="bilinear")(sum)

        elif self.mode == "FCN8s" :
            upsample1 = UpSampling2D(size=(2, 2), interpolation="bilinear")(conv16)
            pool4 = Conv2D(filters=self.num_classes, kernel_size=(1, 1), strides=(1, 1),
                           padding="valid", name="pool4_mod")(pool4)
            sum1 = Add()([upsample1, pool4])
            upsample2 = UpSampling2D(size=(2, 2), interpolation="bilinear")(sum1)
            pool3 = Conv2D(filters=self.num_classes, kernel_size=(1, 1), strides=(1, 1),
                           padding="valid", name="pool3_mod")(pool3)
            sum2 = Add()([upsample2, pool3])
            output = UpSampling2D(size=(8, 8), interpolation="bilinear")(sum2)

        model = keras.Model(inputs=input, outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        return model

    def build_net_pretrained(self):

        input = keras.Input(self.input_shape)
        backbone = VGG16(input_tensor=input, include_top=False)
        backbone.trainable = True
        backbone_feature = backbone.output

        model = keras.Model(inputs=input, outputs=backbone_feature)
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        return model


if __name__ == "__main__" :

    a = FCN(input_shape=(224, 224, 3), num_classes=21, lr=0.001, mode="FCN16s")
    model = a.build_net()
    model.summary()