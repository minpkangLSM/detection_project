import os
import sys
import numpy as np
from PIL import Image
from data_gen import datagen
from DeepResUNet import DeepResUNet

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class InterruptingCallback(tf.keras.callbacks.Callback):

    # ORDER : test begin -> test end -> epoch end
    """
    def on_train_end does not work.
    """

    def on_epoch_end(self, epoch, logs=None):
        print("\nApproached End of epoch.")
        if logs.get("val_updated_mean_io_u") > INITIAL_ACCURACY :
            print("Validation miou goes over initial acc : {0}".format(logs.get("val_updated_mean_io_u")))
            raise InterruptedError("Raised Interrupted Error, satisfying the conditions. Start regenerating training msk.")

def data_renewal(mask, name, folder_dir, renewal_num):

    name = name + ".png"
    dir = os.path.dirname(folder_dir) + "/msk_" + str(renewal_num) + "/img_all"

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

TRAIN_IMG_DIR = "./env_dataset/train/img"
TRAIN_MSK_DIR = "./env_dataset/train/msk"
VALID_IMG_DIR = "./env_dataset/valid/img"
VALID_MSK_DIR = "./env_dataset/valid/msk"
BATCH_SIZE = 6
SEED = 10
INITIAL_EPOCH = 0

valid_data_count = os.listdir(VALID_IMG_DIR+"/"+"img_all")

# Setting
INITIAL_ACCURACY = 0.75
ACCURACY_STEP = 0.015
WEIGHT_DIR = "./semi_test2/weights/weight"
RENEWAL_NUM = 0

# Load Model
DeepResUNet = DeepResUNet(input_size=INPUT_SHAPE,
                          lr=LR,
                          num_classes=NUM_CLASSES)
model = DeepResUNet.build_net()
model.summary()

# START TRAINING
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=WEIGHT_DIR+"/"+"DRU_{epoch:02d}.hdf5"),
             tf.keras.callbacks.TensorBoard(log_dir="./semi_test2/logs", update_freq="batch"),
             InterruptingCallback()]

while True :

    if INITIAL_ACCURACY > 0.85 : break

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

        model.fit(train_gen,
                  steps_per_epoch=36423. / BATCH_SIZE,
                  initial_epoch=INITIAL_EPOCH,
                  validation_data=valid_gen,
                  validation_batch_size=BATCH_SIZE,
                  validation_steps=len(valid_data_count) / BATCH_SIZE,
                  epochs=100,
                  callbacks=[callbacks])

    except :

        # Setting load weights, initial epoch
        weight_list = os.listdir(WEIGHT_DIR)
        last_weight_file = weight_list[-1]
        TRAIN_MSK_DIR = "./env_dataset_msk/train/msk"
        # # initialize weight dir
        # last_weight_dir = "./env_weights/DRU_70.hdf5"
        # INITIAL_EPOCH = int(os.path.splitext(last_weight_file.split("_")[-1])[0])
        RENEWAL_NUM += 1
        # print("\nINITIAL EPOCH : ", INITIAL_EPOCH)

        # Generate new dataset
        pred_img_datagen = ImageDataGenerator()
        pred_img_generator = pred_img_datagen.flow_from_directory(directory=TRAIN_IMG_DIR,
                                                                  batch_size=1,
                                                                  shuffle=False,
                                                                  target_size=INPUT_SHAPE[0:2],
                                                                  class_mode=None)
        print("Image Generating Started. Renewal step : {0}".format(str(RENEWAL_NUM)))
        for data, name in zip(pred_img_generator, pred_img_generator.filenames):

            name = str(name).split('/')[-1].split('\\')[-1].split('.')[0]
            landcover_test = model.predict(
                data,
                batch_size=1,
                verbose=1,
                steps=pred_img_generator.samples / 1.
            )
            data_renewal(landcover_test, name, TRAIN_MSK_DIR, RENEWAL_NUM)

        TRAIN_MSK_DIR = os.path.dirname(TRAIN_MSK_DIR) + "/msk_" + str(RENEWAL_NUM)
        INITIAL_ACCURACY = INITIAL_ACCURACY + ACCURACY_STEP
        print("\nReset. Accuracy is updated : {0} to {1}".format(INITIAL_ACCURACY-ACCURACY_STEP, INITIAL_ACCURACY))
        print("\ntrain mask dir : ", TRAIN_MSK_DIR)
        pass


