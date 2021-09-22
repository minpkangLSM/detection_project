import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import MeanIoU

"""
This network's references.
1) Aldino Rizaldy, et al., Ground and Multi-Class Classification of Airborne Laser Scanner Point Clouds Using Fully Convolutional Networks, Remote Sensing, 2018.
2) 
"""

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

def FCN_block(input, filters, d_rate, block_name, activation="relu") :

    c1 = Conv2D(filters=filters,
                kernel_size=(5,5),
                kernel_initializer="he_normal",
                padding="same",
                dilation_rate=d_rate,
                name=block_name+"_c1")(input)
    b1 = BatchNormalization()(c1)
    r1 = Activation(activation=activation)(b1)

    return r1

class hiearchical_FCN_DK :

    def __init__(self, input_shape, lr):
        self.input_shape = input_shape
        self.lr = lr

    def build_net(self):

        Input = keras.Input(self.input_shape)

        # Main cat.
        N1b1 = FCN_block(input=Input, filters=16, d_rate=1, block_name="N1_b1")
        N1b2 = FCN_block(input=N1b1, filters=32, d_rate=2, block_name="N1_b2")
        N1b3 = FCN_block(input=N1b2, filters=32, d_rate=3, block_name="N1_b3")
        N1b4 = FCN_block(input=N1b3, filters=32, d_rate=4, block_name="N1_b4")
        N1b5 = FCN_block(input=N1b4, filters=32, d_rate=5, block_name="N1_b5")
        N1b6 = FCN_block(input=N1b5, filters=64, d_rate=6, block_name="N1_b6")
        N1_output = Conv2D(filters=8,
                           kernel_size=(1,1),
                           padding="valid",
                           activation="softmax",
                           name="N1_b7_conv2d")(N1b6)

        # Middle cat.
        Input_2 = Concatenate()([Input, N1_output])

        N2b1 = FCN_block(input=Input_2, filters=32, d_rate=1, block_name="N2_b1")
        N2b2 = FCN_block(input=N2b1, filters=64, d_rate=2, block_name="N2_b2")
        N2b3 = FCN_block(input=N2b2, filters=64, d_rate=3, block_name="N2_b3")
        N2b4 = FCN_block(input=N2b3, filters=64, d_rate=4, block_name="N2_b4")
        N2b5 = FCN_block(input=N2b4, filters=64, d_rate=5, block_name="N2_b5")
        N2b6 = FCN_block(input=N2b5, filters=128, d_rate=6, block_name="N2_b6")
        N2_output = Conv2D(filters=23,
                           kernel_size=(1,1),
                           padding="valid",
                           activation="softmax",
                           name="N2_b7_conv2d")(N2b6)

        # Detail cat.
        Input_3 = Concatenate()([Input, N2_output])

        N3b1 = FCN_block(input=Input_3, filters=64, d_rate=1, block_name="N3_b1")
        N3b2 = FCN_block(input=N3b1, filters=128, d_rate=2, block_name="N3_b2")
        N3b3 = FCN_block(input=N3b2, filters=128, d_rate=3, block_name="N3_b3")
        N3b4 = FCN_block(input=N3b3, filters=128, d_rate=4, block_name="N3_b4")
        N3b5 = FCN_block(input=N3b4, filters=128, d_rate=5, block_name="N3_b5")
        N3b6 = FCN_block(input=N3b5, filters=256, d_rate=6, block_name="N3_b6")
        N3_output = Conv2D(filters=42,
                           kernel_size=(1, 1),
                           padding="valid",
                           activation="softmax",
                           name="N3_b7_conv2d")(N3b6)

        model = keras.Model(inputs=Input, outputs=[N1_output, N2_output, N3_output])
        model.compile(
            optimizer = keras.optimizers.Adam(self.lr),
            loss = {"N1_output" : SparseCategoricalCrossentropy(), "N2_output" : SparseCategoricalCrossentropy(), "N3_output" : SparseCategoricalCrossentropy()},
            metrics = {"N1_output" : UpdatedMeanIoU(num_classes=8), "N2_output" : UpdatedMeanIoU(num_classes=23), "N3_output" : UpdatedMeanIoU(num_classes=42)}
        )

        return model

if __name__ == "__main__" :

    INPUT_SHAPE = (1024, 1024, 3)
    LEARNING_RATE  = 0.001
    BATCH_SIZE = 4

    model = hiearchical_FCN_DK(input_shape = INPUT_SHAPE, lr = LEARNING_RATE)
    H_FCN_DK = model.build_net()
    H_FCN_DK.summary()