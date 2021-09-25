import cv2
from utils import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, DepthwiseConv2D, SeparableConv2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Concatenate, Flatten, Dense, BatchNormalization, Activation
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy

class Xception :

    def __init__(self, input_shape, num_classes, lr):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lr = lr

    def build_net(self):

        input = keras.Input(shape=self.input_shape)

        # Entry Flow
        E_conv1 = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding="same")(input)
        E_batch1 = BatchNormalization()(E_conv1)
        E_activ1 = Activation(activation="relu")(E_batch1)

        E_conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(E_activ1)
        E_batch2 = BatchNormalization()(E_conv2)
        E_activ2 = Activation(activation="relu")(E_batch2)

        En_m1 = X_module_entry(input=E_activ2, knum_out=128)
        En_m2 = X_module_entry(input=En_m1, knum_out=256)
        En_m3 = X_module_entry(input=En_m2, knum_out=728)

        # Middle Flow
        M_m1 = X_module_middle(input=En_m3, knum_out=728)
        M_m2 = X_module_middle(input=M_m1, knum_out=728)
        M_m3 = X_module_middle(input=M_m2, knum_out=728)
        M_m4 = X_module_middle(input=M_m3, knum_out=728)
        M_m5 = X_module_middle(input=M_m4, knum_out=728)
        M_m6 = X_module_middle(input=M_m5, knum_out=728)
        M_m7 = X_module_middle(input=M_m6, knum_out=728)
        M_m8 = X_module_middle(input=M_m7, knum_out=728)

        # End Flow
        Ex_m1 = X_module_exit(input=M_m8, knum_out1=728, knum_out2=1024)

        Ex_conv1 = SeparableConv2D(filters=1536, kernel_size=(3,3), strides=(1,1), padding="same")(Ex_m1)
        Ex_batch1 = BatchNormalization()(Ex_conv1)
        Ex_activ1 = Activation(activation="relu")(Ex_batch1)

        Ex_conv2 = SeparableConv2D(filters=2048, kernel_size=(3,3), strides=(1,1), padding="same")(Ex_activ1)
        Ex_batch2 = BatchNormalization()(Ex_conv2)
        Ex_activ2 = Activation(activation="relu")(Ex_batch2)
        GAP = GlobalAveragePooling2D()(Ex_activ2)

        F = Flatten()(GAP)
        D = Dropout(rate=0.5)(F)
        D = Dense(units=self.num_classes, activation="softmax")(D)

        model = keras.Model(inputs=input, outputs=D)

        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss = [CategoricalCrossentropy()],
            metrics = ["accuracy"]
        )

        return model

class Xception_DLV3P :

    def __init__(self, input_shape, num_classes, lr, u_rate):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lr = lr
        self.u_rate = u_rate

    def build_net(self):
        dilation_rate_temp = 2

        input = keras.Input(shape=self.input_shape)

        # Entry flow
        conv1 = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding="same", name="conv1")(input)
        conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", name="conv2")(conv1)

        En1 = X_module_entry_DLV3P(input=conv2, knum=128, u_rate=self.u_rate, c_rate=1, block_name="En1")
        En2 = X_module_entry_DLV3P(input=En1, knum=256, u_rate=self.u_rate, c_rate=1, block_name="En2")
        En3 = X_module_entry_DLV3P(input=En2, knum=728, u_rate=self.u_rate, c_rate=1, block_name="En3")

        # Middle flow
        Mi1 = X_module_middle_DLV3P(input=En3, knum=728, u_rate=self.u_rate, c_rate=1, block_name="Mi1")
        Mi2 = X_module_middle_DLV3P(input=Mi1, knum=728, u_rate=self.u_rate, c_rate=1, block_name="Mi2")
        Mi3 = X_module_middle_DLV3P(input=Mi2, knum=728, u_rate=self.u_rate, c_rate=1, block_name="Mi3")
        Mi4 = X_module_middle_DLV3P(input=Mi3, knum=728, u_rate=self.u_rate, c_rate=1, block_name="Mi4")
        Mi5 = X_module_middle_DLV3P(input=Mi4, knum=728, u_rate=self.u_rate, c_rate=1, block_name="Mi5")
        Mi6 = X_module_middle_DLV3P(input=Mi5, knum=728, u_rate=self.u_rate, c_rate=1, block_name="Mi6")
        Mi7 = X_module_middle_DLV3P(input=Mi6, knum=728, u_rate=self.u_rate, c_rate=1, block_name="Mi7")
        Mi8 = X_module_middle_DLV3P(input=Mi7, knum=728, u_rate=self.u_rate, c_rate=1, block_name="Mi8")
        Mi9 = X_module_middle_DLV3P(input=Mi8, knum=728, u_rate=self.u_rate, c_rate=1, block_name="Mi9")
        Mi10 = X_module_middle_DLV3P(input=Mi9, knum=728, u_rate=self.u_rate, c_rate=1, block_name="Mi10")
        Mi11 = X_module_middle_DLV3P(input=Mi10, knum=728, u_rate=self.u_rate, c_rate=1, block_name="Mi11")
        Mi12 = X_module_middle_DLV3P(input=Mi11, knum=728, u_rate=self.u_rate, c_rate=1, block_name="Mi12")
        Mi13 = X_module_middle_DLV3P(input=Mi12, knum=728, u_rate=self.u_rate, c_rate=1, block_name="Mi13")
        Mi14 = X_module_middle_DLV3P(input=Mi13, knum=728, u_rate=self.u_rate, c_rate=1, block_name="Mi14")
        Mi15 = X_module_middle_DLV3P(input=Mi14, knum=728, u_rate=self.u_rate, c_rate=1, block_name="Mi15")
        Mi16 = X_module_middle_DLV3P(input=Mi15, knum=728, u_rate=self.u_rate, c_rate=1, block_name="Mi16")

        # Exit flow
        Ex1 = X_module_exit_DLV3P(input=Mi16, knum_in=728, knum_out=1024, u_rate=self.u_rate, c_rate=1, block_name="Ex1")
        Ex1_conv1 = SeparableConv2D(filters=1536, kernel_size=(3,3), kernel_initializer="he_normal",
                                    strides=(1,1), dilation_rate=dilation_rate_temp, padding="same", name="Ex1_conv1")(Ex1)
        Ex1_batch1 = BatchNormalization()(Ex1_conv1)
        Ex1_active1 = Activation(activation="relu", name="Ex1_active1")(Ex1_batch1)

        Ex1_conv2 = SeparableConv2D(filters=1536, kernel_size=(3,3), kernel_initializer="he_normal",
                                    strides=(1,1), dilation_rate=dilation_rate_temp, padding="same", name="Ex1_conv2")(Ex1_active1)
        Ex1_batch2 = BatchNormalization()(Ex1_conv2)
        Ex1_active2 = Activation(activation="relu", name="Ex1_active2")(Ex1_batch2)

        Ex1_conv3 = SeparableConv2D(filters=2048, kernel_size=(3,3), kernel_initializer="he_normal",
                                    strides=(1,1), dilation_rate=dilation_rate_temp, padding="same", name="Ex1_conv3")(Ex1_active2)
        Ex1_batch3 = BatchNormalization()(Ex1_conv3)
        Ex1_active3 = Activation(activation="relu", name="Ex1_active3")(Ex1_batch3)

        GAP = GlobalAveragePooling2D()(Ex1_active3)

        F = Flatten()(GAP)
        D = Dropout(rate=0.5)(F)
        D = Dense(units=self.num_classes, activation="softmax")(D)

        model = keras.Model(inputs=input, outputs=D)

        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=[CategoricalCrossentropy()],
            metrics=["accuracy"]
        )

        return model

if __name__ == "__main__" :

    INPUT_SHAPE = (512, 512, 3)
    NUM_CLASSES = 100
    LEARNING_RATE = 0.001
    UNIT_RATE = (1,2,4)

    DLV3P = Xception_DLV3P(INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE, UNIT_RATE)
    model = DLV3P.build_net()
    model.summary()