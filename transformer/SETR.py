"""
2021.03.30. ~
Implementation of "Rethinkng Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers"(SETR), Sixaio Zheng, etal. 31, Dec, 2020.
by Kangmin Park. Lab for Sensor and Modeling, Geoinformatics, Univ. of Seoul.
"""
import os
os.path.join("..")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

def scheduler(epoch, lr):

    initial_lr = 0.00001
    end_lr = 0.0000001
    decay_step = 100
    lr = (initial_lr-end_lr)*(1-epoch/decay_step)+end_lr
    return lr

class MyCallBack(Callback):

    def on_epoch_end(self, epoch, logs=None):
        print(self.model.optimizer.lr)

def position_embedding(shape, initializer):
    pe = tf.Variable(initializer(shape=shape), dtype=tf.float32)
    return pe.numpy()

def sequentializing_image(input_tensor,
                          size,
                          embedding_dim):
    """
    Sequentialize an image based on "size" value.
    :param input_tensor: an input image, shape : [batch, height, width, channels=3]
    :param ratio : ratio of split size
    :param embedding_dim : dimension of embedding tensor
    :return: need to check
    """
    # extract_patches returns patches flatten. check the shape.
    patches = tf.image.extract_patches(images=input_tensor,
                                       sizes=[1, size, size, 1],
                                       strides=[1, size, size, 1],
                                       rates=[1, 1, 1, 1],
                                       padding="VALID")
    # convert sequentialized image features into dimensionality of R^{C}
    patches_reshaped = tf.reshape(patches,
                                  shape=[-1, patches.shape[1]*patches.shape[2], patches.shape[3]])
    # Embedding to R^C
    # embedding = Dense(units=embedding_dim, name="seq_dense")(patches_reshaped)
    embedding = Conv1D(filters=embedding_dim, kernel_size=1, strides=1, padding="valid")(patches_reshaped)
    # naive positional encoding
    position_em = position_embedding(shape=(1, embedding.shape[1], embedding.shape[2]),
                                     initializer=tf.random_normal_initializer())
    return embedding + position_em

def multi_self_attention(input_tensor,
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

    dn1 = Dense(units=hidden_size)(input_tensor)
    dp = Dropout(rate=0.2)(dn1)
    gl = Activation(activation="gelu")(dp)
    dn2 = Dense(units=hidden_size)(gl)

    return dn2

def Transformer(input_tensor,
                num_heads):

    hidden_size = input_tensor.shape[-1]
    # Key, Query, Value
    ly1 = LayerNormalization()(input_tensor)
    mha = multi_self_attention(input_tensor=ly1,
                               num_of_head=num_heads,
                               hidden_size=hidden_size)
    rb1 = ly1 + mha

    ly2 = LayerNormalization()(rb1)
    mlp = MLP(input_tensor=ly2,
              hidden_size=hidden_size)
    rb2 = mlp + rb1

    return rb2

def decoder_naive(input_tensor,
                  input_shape,
                  size,
                  num_classes) :

    input_tensor = tf.reshape(input_tensor,
                              [-1, int(input_shape[0]/size), int(input_shape[1]/size), input_tensor.shape[-1]])
    conv1 = Conv2D(filters=512,
                   kernel_size=1,
                   strides=1,
                   padding="same")(input_tensor)
    batch1 = BatchNormalization()(conv1)
    conv2 = Conv2D(filters=num_classes,
                   kernel_size=1,
                   strides=1,
                   padding="same")(batch1)
    upsampling = UpSampling2D(size=(size,size),
                              interpolation="bilinear")(conv2)

    return upsampling

def decoder_pup(input_tensor,
                input_shape,
                size,
                num_classes):

    input_tensor = tf.reshape(input_tensor,
                              [-1, int(input_shape[0]/size), int(input_shape[1]/size), input_tensor.shape[-1]])
    conv1 = Conv2DTranspose(filters=256,
                            kernel_size=3,
                            strides=2,
                            padding="same")(input_tensor)
    batch1 = BatchNormalization()(conv1)
    active1 = Activation(activation="relu")(batch1)

    conv2 = Conv2DTranspose(filters=256,
                            kernel_size=3,
                            strides=2,
                            padding="same")(active1)
    batch2 = BatchNormalization()(conv2)
    active2 = Activation(activation="relu")(batch2)

    conv3 = Conv2DTranspose(filters=256,
                            kernel_size=3,
                            strides=2,
                            padding="same")(active2)
    batch3 = BatchNormalization()(conv3)
    active3 = Activation(activation="relu")(batch3)

    conv4 = Conv2DTranspose(filters=num_classes,
                            kernel_size=3,
                            strides=2,
                            padding="same")(active3)
    return conv4

# test
class SETR :

    def __init__(self, INPUT_SHAPE, NUM_CLASSES, LR, RATIO_SIZE, NUM_Tr, NUM_Head, EMBEDDING_DIM=1024):

        self.INPUT_SHAPE = INPUT_SHAPE
        self.NUM_CLASSES = NUM_CLASSES
        self.LR = LR
        self.RATIO_SIZE = RATIO_SIZE

        self.NUM_Tr = NUM_Tr
        self.NUM_Head = NUM_Head
        self.EMBEDDING_DIM = EMBEDDING_DIM

    def build_net(self):

        input = tf.keras.Input(shape=self.INPUT_SHAPE)

        Tr = Conv2D(filters=self.EMBEDDING_DIM,
                    kernel_size=16,
                    strides=16,
                    padding="valid")(input)
        Tr = tf.reshape(Tr, shape=[-1, Tr.shape[1]*Tr.shape[2], Tr.shape[3]])
        pe = position_embedding(shape=[1, Tr.shape[1], Tr.shape[2]],
                                initializer=tf.zeros_initializer())
        Tr = Tr + pe

        for idx in range(self.NUM_Tr) :
            Tr = Transformer(input_tensor=Tr,
                             num_heads=12)

        # decoder = decoder_naive(input_tensor=Tr,
        #                         input_shape=self.INPUT_SHAPE,
        #                         size=self.RATIO_SIZE,
        #                         num_classes=self.NUM_CLASSES)

        decoder = decoder_pup(input_tensor=Tr,
                              input_shape=self.INPUT_SHAPE,
                              size=self.RATIO_SIZE,
                              num_classes=self.NUM_CLASSES)
        output = Activation(activation="softmax")(decoder)

        model = tf.keras.Model(inputs=input, outputs=output)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.LR),
            loss=SparseCategoricalCrossentropy(),
            metrics=UpdatedMeanIoU(num_classes=self.NUM_CLASSES)
        )

        return model

if __name__ == "__main__" :

    INPUT_SHAPE = (512, 512, 3)
    NUM_CLASSES = 2
    LR = 0.00001
    RATIO_SIZE = 16
    NUM_Tr = 12
    NUM_Head = 8
    EMBEDDING_DIM = 768
    BATCH_SIZE = 4
    seed = 10

    # train
    image_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True
    )
    mask_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True
    )

    image_generator = image_datagen.flow_from_directory(
       # directory="./renewal_dataset_0324/train/images",
       directory="./swham/train/images",
       class_mode=None,
       seed=seed,
       shuffle=True,
       target_size=(512, 512),
       batch_size=BATCH_SIZE
    )
    mask_generator = mask_datagen.flow_from_directory(
       #directory="./renewal_dataset_0324/train/labels",
       directory="./swham/train/annotations",
       class_mode=None,
       seed=seed,
       shuffle=True,
       target_size=(512, 512),
       color_mode="grayscale",
       batch_size=BATCH_SIZE
    )

    # valid
    valid_img_datagen = ImageDataGenerator()
    valid_msk_datagen = ImageDataGenerator()

    valid_img_generator = valid_img_datagen.flow_from_directory(
        #directory="./renewal_dataset_0324/valid/images",
        directory="./swham/val/images",
        target_size=(512, 512),
        seed=seed,
        shuffle=True,
        class_mode=None,
        batch_size=BATCH_SIZE
    )
    valid_mask_generator = valid_msk_datagen.flow_from_directory(
        #directory="./renewal_dataset_0324/valid/labels",
        directory="./swham/val/annotations",
        target_size=(512, 512),
        seed=seed,
        shuffle=True,
        class_mode=None,
        color_mode="grayscale",
        batch_size=BATCH_SIZE
    )

    def create_train_generator(img, label):
        while True:
            for x1, x2 in zip(img, label):
                yield x1, x2

    train_gen = create_train_generator(image_generator, mask_generator)
    valid_gen = create_train_generator(valid_img_generator, valid_mask_generator)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath="./setr_pretrained/with_ml_decay/weight/SETR_pretrained_{epoch:02d}.hdf5"),
        tf.keras.callbacks.TensorBoard(log_dir="./setr_pretrained/with_mlp_decay/logs"),
        tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
        MyCallBack()
    ]

    inst = SETR(INPUT_SHAPE=INPUT_SHAPE,
                NUM_CLASSES=NUM_CLASSES,
                LR=LR,
                RATIO_SIZE=RATIO_SIZE,
                NUM_Tr=NUM_Tr,
                NUM_Head=NUM_Head,
                EMBEDDING_DIM=EMBEDDING_DIM)

    model = inst.build_net()
    model.summary()

    model.fit(
        train_gen,
        batch_size=BATCH_SIZE,
        validation_data=valid_gen,
        validation_batch_size=BATCH_SIZE,
        validation_steps=valid_img_generator.samples/BATCH_SIZE,
        steps_per_epoch=mask_generator.samples/BATCH_SIZE,
        epochs=100,
        callbacks=[callbacks]
    )