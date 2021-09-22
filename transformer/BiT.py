"""
2021.06.17.
Kangmin Park, Lab. for Sensor and Modeling, Univ. of Seoul.
Hao Chen, et al., <Efficient Transformer based Method for Remote Sensing Image Change Detection>, 2021.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
        y_pred = tf.math.argmax(y_pred, axis=-1)
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

# Semantic Tokenizer
class Tokenizer :

    @staticmethod
    def Resblock(input,
                 knum,
                 layer_name,
                 pad="same",
                 verbose=False,
                 exception=False):

        # identity mapping
        identity = input

        if verbose:
            if not exception :
                identity = MaxPool2D(pool_size=1, strides=2)(identity)
            zero_pad = K.zeros_like(identity)
            identity = Concatenate()([identity, zero_pad])

        if not verbose :
            Conv_L1 = Conv2D(filters=knum, kernel_size=3, kernel_initializer="he_normal",
                             strides=1, padding=pad, name=layer_name + "_C_L1")(input)
        else :
            if exception :
                Conv_L1 = Conv2D(filters=knum, kernel_size=3, kernel_initializer="he_normal",
                                 strides=1, padding=pad, name=layer_name + "_C_L1")(input)
            else :
                Conv_L1 = Conv2D(filters=knum, kernel_size=3, kernel_initializer="he_normal",
                                 strides=2, padding=pad, name=layer_name + "_C_L1")(input)

        BN_L1 = BatchNormalization()(Conv_L1)
        AC_L1 = Activation(activation="relu")(BN_L1)

        Conv_L2 = Conv2D(filters=knum, kernel_size=3, kernel_initializer="he_normal",
                         strides=1, padding=pad, name=layer_name + "_C_L2")(AC_L1)
        BN_L2 = BatchNormalization()(Conv_L2)

        # shortcut
        shortcut = Add()([BN_L2, identity])
        shortcut = Activation(activation="relu")(shortcut)

        return shortcut

    def resnet18_5s(self,
                    input_shape):

        input = keras.Input(shape=input_shape)

        # conv1 output : 1 -> 1/2
        conv1 = Conv2D(filters=64,
                       kernel_size=7,
                       strides=2,
                       activation="relu",
                       padding="same",
                       name="conv1")(input)
        # pool1 output : 1/2 -> 1/4
        pool1 = MaxPool2D(pool_size=3,
                          strides=2,
                          padding="same",
                          name="pool1")(conv1)

        # rb1 output : 1/4 -> 1/4
        rb1 = Tokenizer.Resblock(input=pool1,
                                 knum=64,
                                 layer_name="rb1")
        # rb2 output : 1/4 -> 1/4
        rb2 = Tokenizer.Resblock(input=rb1,
                                 knum=64,
                                 layer_name="rb2")

        # rb3 output : 1/4 -> 1/8
        rb3 = Tokenizer.Resblock(input=rb2,
                                 knum=128,
                                 layer_name="rb3",
                                 verbose=True)
        # rb4 output : 1/8 -> 1/8
        rb4 = Tokenizer.Resblock(input=rb3,
                                 knum=128,
                                 layer_name="rb4")
        # rb5 output : 1/8 -> 1/8
        rb5 = Tokenizer.Resblock(input=rb4,
                                 knum=256,
                                 layer_name="rb5",
                                 verbose=True,
                                 exception=True)
        # rb6 output : 1/8 -> 1/8
        rb6 = Tokenizer.Resblock(input=rb5,
                                 knum=256,
                                 layer_name="rb6")

        # rb7 output : 1/8 -> 1/8
        rb7 = Tokenizer.Resblock(input=rb6,
                                 knum=512,
                                 layer_name="rb7",
                                 verbose=True,
                                 exception=True)
        # rb8 output : 1/8 -> 1/8
        rb8 = Tokenizer.Resblock(input=rb7,
                                 knum=512,
                                 layer_name="rb8")

        usp = UpSampling2D(size=2,
                           interpolation="bilinear")(rb8)
        conv2 = Conv2D(filters=32,
                       kernel_size=1,
                       strides=1,
                       padding="valid")(usp)

        model = keras.Model(inputs=input,
                            outputs=conv2)

        return model

    def Token(self,
              input1,
              input2,
              input_shape,
              token_length):

        resnet = Tokenizer().resnet18_5s(input_shape=input_shape)

        x1 = resnet(input1)
        x2 = resnet(input2)
        x1_shape = x1.shape
        x1 = tf.reshape(x1,
                        shape=[-1, x1.shape[1] * x1.shape[2], x1.shape[3]])
        x2 = tf.reshape(x2,
                        shape=[-1, x2.shape[1] * x2.shape[2], x2.shape[3]])
        # shape of x1, x2 : [batch, H*W, Token_length]
        A1 = tf.keras.activations.softmax(Dense(units=token_length)(x1), axis=1)
        A2 = tf.keras.activations.softmax(Dense(units=token_length)(x2), axis=1)
        A1 = tf.transpose(A1, perm=[0, 2, 1]) # [batch, Token_length, H*W]
        A2 = tf.transpose(A2, perm=[0, 2, 1]) # [batch, Token_length, H*W]

        T1 = tf.matmul(A1, x1) # [batch, Token_length, C]
        T2 = tf.matmul(A2, x2) # [batch, Token_length, C]

        return T1, T2, x1, x2, x1_shape

# Transformer Encoder
def position_embedding(shape, initializer):
    pe = tf.Variable(initializer(shape=shape[1:]), dtype=tf.float32)
    return pe.numpy()

def MSA(input_tensor,
        hidden_size,
        num_of_head):

    projection_dim = hidden_size//num_of_head

    # query
    query = Dense(units=hidden_size)(input_tensor)
    multi_head_query = tf.reshape(query,
                                  shape=[-1, query.shape[1], num_of_head, projection_dim])
    multi_head_query = tf.transpose(multi_head_query,
                                    perm=[0, 2, 1, 3]) # [batch, num_of_head, seq_length, project_dim]

    # key
    key = Dense(units=hidden_size)(input_tensor)
    multi_head_key = tf.reshape(key,
                                shape=[-1, key.shape[1], num_of_head, projection_dim])
    multi_head_key = tf.transpose(multi_head_key,
                                  perm=[0, 2, 1, 3]) # [batch, num_of_head, seq_length, project_dim]

    # value
    value = Dense(units=hidden_size)(input_tensor)
    multi_head_value = tf.reshape(value,
                                  shape=[-1, value.shape[1], num_of_head, projection_dim])
    multi_head_value = tf.transpose(multi_head_value,
                                    perm=[0, 2, 1, 3]) # [batch, num_of_head, seq_length, project_dim]
    score = tf.matmul(multi_head_query, multi_head_key, transpose_b=True) # [batch, num_of_head, seq_length, seq_length]
    scale = tf.cast(tf.shape(key)[-1], score.dtype)
    scaled_score = score / tf.math.sqrt(scale)
    attention = tf.keras.activations.softmax(scaled_score, axis=-1) # [batch, num_of_head, seq_length, seq_length]
    output = tf.matmul(attention, multi_head_value) # [batch, num_of_head, seq_length, project_dim]
    output = tf.transpose(output,
                          perm=[0, 2, 1, 3])

    # output = tf.transpose(output, [0, 2, 1, 3])
    concat_output = tf.reshape(output,
                               shape=[-1, output.shape[1], hidden_size])
    # concat dense
    output = Dense(units=hidden_size)(concat_output)

    return output

def MLP(input_tensor,
        hidden_size):

    dn1 = Dense(units=2*hidden_size)(input_tensor)
    dp = Dropout(rate=0.5)(dn1)
    gl = Activation(activation="gelu")(dp)
    dn2 = Dense(units=hidden_size)(gl)

    return dn2

def Transformer_encoder(input_tensor,
                        num_heads):

    hidden_size = input_tensor.shape[-1]
    # Key, Query, Value
    ly1 = LayerNormalization()(input_tensor)
    msa = MSA(input_tensor=ly1,
              num_of_head=num_heads,
              hidden_size=hidden_size)
    rb1 = ly1 + msa

    ly2 = LayerNormalization()(rb1)
    mlp = MLP(input_tensor=ly2,
              hidden_size=hidden_size)
    rb2 = mlp+rb1

    return rb2

# Transformer Decoder
def MA(query,
       value,
       num_of_head):
    """

    :param query: X^i
    :param value: L^i_new
    :param num_of_head:
    :return:
    """

    hidden_size = query.shape[-1]
    projection_dim = hidden_size // num_of_head

    # query : x
    query = Dense(units=hidden_size)(query)
    multi_head_query = tf.reshape(query,
                                  shape=[-1, query.shape[1], num_of_head, projection_dim])
    multi_head_query = tf.transpose(multi_head_query,
                                    perm=[0, 2, 1, 3])  # [batch, num_of_head, query.shape[1](sequential_length), projection_dim]

    # key / value : t
    key = Dense(units=hidden_size)(value)
    multi_head_key = tf.reshape(key,
                                shape=[-1, value.shape[1], num_of_head, projection_dim])
    multi_head_key = tf.transpose(multi_head_key,
                                  perm=[0, 2, 1, 3])  # [batch, num_of_head, value.shape[1](sequential_length), projection_dim]

    value = Dense(units=hidden_size)(value)
    multi_head_value = tf.reshape(value,
                                  shape=[-1, value.shape[1], num_of_head, projection_dim])
    multi_head_value = tf.transpose(multi_head_value,
                                    perm=[0, 2, 1, 3])  # [batch, num_of_head, value.shape[1](sequential_length), projection_dim]

    # attention
    A = tf.matmul(multi_head_query, multi_head_key,
                  transpose_b=True)  # A = [batch_size, num_of_head, query.shape[1], value.shape[1]]
    scale = tf.cast(tf.shape(key)[-1], A.dtype)
    scaled_A = A / tf.math.sqrt(scale)
    A = tf.keras.activations.softmax(scaled_A, axis=-1)
    output = tf.matmul(A, multi_head_value)  # [batch_size, num_of_head, query.shape[1], projection_dim]
    output = tf.transpose(output,
                          perm=[0, 2, 1, 3])
    concat_output = tf.reshape(output,
                               shape=[-1, output.shape[1], hidden_size])
    output = Dense(units=hidden_size)(concat_output)

    return output

def Transformer_decoder(x,
                        t,
                        num_heads):

    hidden_size = x.shape[-1]
    # Key, Query, Value
    ly1 = LayerNormalization()(x)
    ma = MA(query=x,
            value=t,
            num_of_head=num_heads)
    rb1 = ly1 + ma

    ly2 = LayerNormalization()(rb1)
    mlp = MLP(input_tensor=ly2,
              hidden_size=hidden_size)
    rb2 = mlp+rb1

    return rb2

# main
class BiT :

    def __init__(self,
                 input_shape,
                 L,
                 decoder_num,
                 num_classes):

        self.input_shape = input_shape
        self.L = L
        self.decoder_num = decoder_num
        self.num_classes = num_classes

    def build_net(self):

        input1 = keras.Input(shape=self.input_shape)
        input2 = keras.Input(shape=self.input_shape)

        """---Tokenizer---"""
        token_inst = Tokenizer()
        t1, t2, x1, x2, x_shape = token_inst.Token(input1=input1,
                                                   input2=input2,
                                                   input_shape=self.input_shape,
                                                   token_length=self.L)
        """---Encoding---"""
        # shape of t1, t2 : [batch, token_length, C] -> concat axis=1 : [batch, 2token_length, C]
        t = Concatenate(axis=1, name="Concat_encoding")([t1, t2])
        # positional encoding
        pe = position_embedding(shape=list(t.shape), initializer=tf.random_normal_initializer())
        t = t+pe
        t_new = Transformer_encoder(input_tensor=t,
                                    num_heads=8)

        """---Decoder---"""
        t1_new, t2_new = t_new[:, :t_new.shape[1]//2, :], t_new[:, t_new.shape[1]//2:, :]
        for idx in range(self.decoder_num) :
            x1 = Transformer_decoder(x=x1,
                                     t=t1_new,
                                     num_heads=8)
            x2 = Transformer_decoder(x=x2,
                                     t=t2_new,
                                     num_heads=8)

        """---prediction---"""
        x1 = tf.reshape(x1,
                        shape=[-1, x_shape[1], x_shape[2], x_shape[3]])
        x2 = tf.reshape(x2,
                        shape=[-1, x_shape[1], x_shape[2], x_shape[3]])
        x1 = UpSampling2D(size=4,
                          interpolation="bilinear")(x1)
        x2 = UpSampling2D(size=4,
                          interpolation="bilinear")(x2)
        x_fdi = x1-x2
        x_fdi = Conv2D(filters=2,
                       kernel_size=3,
                       strides=1,
                       padding="same")(x_fdi)
        output = Activation(activation="softmax")(x_fdi)

        model = keras.Model(inputs=(input1, input2),
                            outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=SparseCategoricalCrossentropy(),
            metrics=UpdatedMeanIoU(num_classes=self.num_classes)
        )

        return model

if __name__ == "__main__" :

    input_shape = (512, 512, 3)
    L = 8
    decoder_num = 8
    num_classes=2
    BATCH_SIZE=2
    seed=10
    LR = 0.00001

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
        rescale=1./255
    )

    image_generator_2019 = image_datagen_2019.flow_from_directory(
        directory="./seoul_cd_210624/train/A",
        class_mode=None,
        seed=seed,
        shuffle=True,
        target_size=(512, 512),
        batch_size=BATCH_SIZE
    )
    image_generator_2018 = image_datagen_2018.flow_from_directory(
        directory="./seoul_cd_210624/train/B",
        class_mode=None,
        seed=seed,
        shuffle=True,
        target_size=(512, 512),
        batch_size=BATCH_SIZE
    )
    mask_generator = mask_datagen.flow_from_directory(
        directory="./seoul_cd_210624/train/cmap",
        class_mode=None,
        seed=seed,
        shuffle=True,
        target_size=(512, 512),
        color_mode="grayscale",
        batch_size=BATCH_SIZE
    )

    # valid
    image_datagen_2019 = ImageDataGenerator()
    image_datagen_2018 = ImageDataGenerator()
    mask_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    valid_image_generator_2019 = image_datagen_2019.flow_from_directory(
        directory="./seoul_cd_210624/val/A",
        class_mode=None,
        seed=seed,
        shuffle=True,
        target_size=(512, 512),
        batch_size=BATCH_SIZE
    )
    valid_image_generator_2018 = image_datagen_2018.flow_from_directory(
        directory="./seoul_cd_210624/val/B",
        class_mode=None,
        seed=seed,
        shuffle=True,
        target_size=(512, 512),
        batch_size=BATCH_SIZE
    )
    valid_mask_generator = mask_datagen.flow_from_directory(
        directory="./seoul_cd_210624/val/cmap",
        class_mode=None,
        seed=seed,
        shuffle=True,
        target_size=(512, 512),
        color_mode="grayscale",
        batch_size=BATCH_SIZE
    )


    def create_pred_generator(img1, img2, label):
        while True:
            for x1, x2, x3 in zip(img1, img2, label):
                yield (x1, x2), x3

    inst = BiT(input_shape=input_shape,
               L=L,
               decoder_num=decoder_num,
               num_classes=num_classes)
    model = inst.build_net()
    model.summary()

    train_gen = create_pred_generator(image_generator_2018, image_generator_2019, mask_generator)
    valid_gen = create_pred_generator(valid_image_generator_2018, valid_image_generator_2019, valid_mask_generator)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath="./BiT_seoulcd/weights/BiT_{epoch:02d}.hdf5"),
        tf.keras.callbacks.TensorBoard(log_dir="./BiT_seoulcd/logs", update_freq="batch"),
        tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
        MyCallBack()
    ]

    model.fit(
        train_gen,
        batch_size=BATCH_SIZE,
        validation_data=valid_gen,
        validation_batch_size=BATCH_SIZE,
        validation_steps=valid_image_generator_2019.samples / BATCH_SIZE,
        steps_per_epoch=mask_generator.samples * 2 // BATCH_SIZE,
        epochs=100,
        callbacks=[callbacks]
    )