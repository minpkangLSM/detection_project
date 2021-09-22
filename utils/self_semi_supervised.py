import os
import sys
import numpy as np
from PIL import Image
sys.path.append("..")
from utils.data_gen import datagen
from segmentations.DeepResUNet import DeepResUNet

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# class SaveWeights(tf.keras.callbacks.ModelCheckpoint):
#     print("Save Point.")

class InterruptingCallback(tf.keras.callbacks.Callback):

    # ORDER : https://www.notion.so/How-to-set-threshold-of-acc-in-Keras-cd0a4bc09587426ea9b7c36b62381f97

    def on_train_begin(self, logs=None):
        print("\nOn train begin")

    def on_train_end(self, logs=None):
        print("\nOn train end")

    def on_train_batch_end(self, batch, logs={}):
        print("\nOn train batch end")

    def on_train_batch_begin(self, batch, logs={}):
        print("\nOn train batch begin")

    def on_batch_begin(self, batch, logs={}):
        print("\nOn batch begin")

    def on_batch_end(self, batch, logs={}):
        print("\nOn batch end")

    def on_epoch_begin(self, epoch, lgos={}):
        print("\nOn epoch begin")

    def on_epoch_end(self, epoch, logs=None):
        print("\nOn epoch end.")
        # if logs.get("val_updated_mean_io_u") > INITIAL_ACCURACY :
        #     print("test miou is over initial acc : {0}".format(logs.get("val_updated_mean_io_u")))
        #     raise InterruptedError("Raised Interrupted Error.")

    def on_test_batch_begin(self, batch, logs=None):
        print("\nOn test batch begin")

    def on_test_batch_end(self, batch, logs=None):
        print("\nOn test batch end")

    def on_test_begin(self, logs={}):
        print("\nOn test begin")

    def on_test_end(self, logs={}):
        print("\nOn test end.")

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def data_renewal(mask, name, folder_dir, renewal_num):

    name = name + ".png"
    dir = os.path.dirname(folder_dir) + "\\S_labels_" + str(renewal_num) + "\\img_all"

    if not os.path.exists(dir) :
        os.makedirs(dir)

    mask = mask.reshape(512, 512, 2)
    mask = np.argmax(mask, axis=-1)
    im = Image.fromarray(mask.astype(np.uint8))
    im.save(dir+"/"+name, 'png')

# Model Parameters
INPUT_SHAPE = (512, 512, 3)
NUM_CLASSES = 2
LR = 1e-4

TRAIN_IMG_DIR = "E:\\2020_SeoulChangeDetection\\Data\\test_images\\test_img\\clip"
TRAIN_MSK_DIR = "E:\\2020_SeoulChangeDetection\\Data\\test_images\\test_msk\\z\\S_labels"
VALID_IMG_DIR = "E:\\2020_SeoulChangeDetection\\Data\\test_images\\test_img\\clip"
VALID_MSK_DIR = "E:\\2020_SeoulChangeDetection\\Data\\test_images\\test_msk\\z\\S_labels"
BATCH_SIZE = 6
SEED = 10
INITIAL_EPOCH = 0

valid_data_count = os.listdir(VALID_IMG_DIR+"/"+"all_images")

# Setting
INITIAL_ACCURACY = 0.75
ACCURACY_STEP = 0.015
WEIGHT_DIR = "E:\\deep_learning_project\\weights\\weight"
RENEWAL_NUM = 0

# Load Model
DeepResUNet = DeepResUNet(input_size=INPUT_SHAPE,
                          lr=LR,
                          num_classes=NUM_CLASSES)
model = DeepResUNet.build_net()
model.load_weights("E:\\deep_learning_project\\weights\\DRU_70.hdf5")

# START TRAINING
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=WEIGHT_DIR+"\\"+"DRU_{epoch:02d}.hdf5"),
             #tf.keras.callbacks.TensorBoard(log_dir=".\\semi_test\\logs", update_freq="batch"),
             InterruptingCallback()]

while True :

    if INITIAL_ACCURACY > 0.0 : break

    train_gen = datagen(img_dir=TRAIN_IMG_DIR,
                        msk_dir=TRAIN_MSK_DIR,
                        input_size=512,
                        batch=BATCH_SIZE,
                        seed=SEED,
                        horizon=True,
                        vertical=True,
                        shuffle=True)

    valid_gen = datagen(img_dir=VALID_IMG_DIR,
                        msk_dir=VALID_MSK_DIR,
                        input_size=512,
                        batch=BATCH_SIZE,
                        seed=SEED,
                        horizon=False,
                        vertical=False,
                        shuffle=False)

    try :

        if INITIAL_EPOCH != 0 :
            model.load_weights(last_weight_dir)

        model.fit(train_gen,
                  steps_per_epoch=3,
                  initial_epoch=INITIAL_EPOCH,
                  validation_data=valid_gen,
                  validation_batch_size=BATCH_SIZE,
                  validation_steps=2,
                  epochs=1,
                  callbacks=[callbacks])

    except :

        # Setting load weights, initial epoch
        weight_list = os.listdir(WEIGHT_DIR)
        last_weight_file = weight_list[-1]
        last_weight_dir = os.path.join(WEIGHT_DIR, last_weight_file)
        INITIAL_EPOCH = int(os.path.splitext(last_weight_file.split("_")[-1])[0])
        print("\nINITIAL EPOCH : ", INITIAL_EPOCH)
        RENEWAL_NUM += 1

        # Generate new dataset
        pred_img_datagen = ImageDataGenerator()
        pred_img_generator = pred_img_datagen.flow_from_directory(directory=TRAIN_IMG_DIR,
                                                                  batch_size=1,
                                                                  target_size=INPUT_SHAPE[0:2],
                                                                  shuffle=False,
                                                                  class_mode=None)
        for data, name in zip(pred_img_generator, pred_img_generator.filenames):

            name = str(name).split('/')[-1].split('\\')[-1].split('.')[0]
            landcover_test = model.predict(
                data,
                batch_size=1,
                verbose=1,
                steps=pred_img_generator.samples / 1.
            )
            data_renewal(landcover_test, name, TRAIN_MSK_DIR, RENEWAL_NUM)

        print("RENEWAL_NUM : ", RENEWAL_NUM)
        TRAIN_MSK_DIR = os.path.dirname(TRAIN_MSK_DIR) + "\\S_labels_" + str(RENEWAL_NUM)
        INITIAL_ACCURACY = INITIAL_ACCURACY + ACCURACY_STEP
        print("\nReset. Accuracy is updated : {0} to {1}".format(INITIAL_ACCURACY-ACCURACY_STEP, INITIAL_ACCURACY))

        print("train mask dir : ", TRAIN_MSK_DIR)
        pass


