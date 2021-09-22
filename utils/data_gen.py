import os
import sys
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def datagen(img_dir,
            msk_dir,
            input_size,
            batch,
            seed=10,
            horizon=False,
            vertical=False,
            rotation=0,
            shuffle=False):

    IMG_GENERATOR_SET = ImageDataGenerator(
        horizontal_flip=horizon,
        vertical_flip=vertical,
        rotation_range=rotation
    )
    MSK_GENERATOR_SET = ImageDataGenerator(
        horizontal_flip=horizon,
        vertical_flip=vertical,
        rotation_range=rotation
    )
    IMG_GENERATOR = IMG_GENERATOR_SET.flow_from_directory(
        directory=img_dir,
        class_mode=None,
        seed=seed,
        shuffle=shuffle,
        target_size=(input_size, input_size),
        batch_size=batch
    )
    MSK_GENERATOR = MSK_GENERATOR_SET.flow_from_directory(
        directory=msk_dir,
        class_mode=None,
        seed=seed,
        shuffle=shuffle,
        target_size=(input_size, input_size),
        batch_size=batch,
        color_mode="grayscale"
    )

    DATA_SET = zip(IMG_GENERATOR, MSK_GENERATOR)

    return DATA_SET

if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description="Tensorflow ImageDataGenerator")

    parser.add_argument("--img_dir", "-id")
    parser.add_argument("--msk_dir", "-md")
    parser.add_argument("--seed", "-s",
                        type=int,
                        default=10,
                        help="initialize seed info.")
    parser.add_argument("--batch", "-b",
                        type=int,
                        default=1,
                        help="initialize batch size")
    parser.add_argument("--input_size", "-i",
                        type=int,
                        help="input size, width and height are identical.")
    # data augmentation options
    parser.add_argument("--horizon", "-ho",
                        action="store_true",
                        help="data augmentation : horizontal flip")
    parser.add_argument("--vertical", "-ve",
                        action="store_true",
                        help="data augmentation : vertical flip")
    parser.add_argument("--rotation", "-ro",
                        type=float,
                        default=0,
                        help="data augmentation : rotation angle")
    parser.add_argument("--shuffle", "-sh",
                        action="store_true",
                        help="data augmentation : setting input order random")

    args = parser.parse_args()
    print(args.seed)
    print(args.batch)

    def datagen(img_dir=args.img_dir,
                msk_dir=args.msk_dir,
                seed=args.seed,
                batch=args.batch,
                input_size=args.input_size,
                horizon=args.horizon,
                vertical=args.vertical,
                rotation=args.rotation,
                shuffle=args.shuffle):

        IMG_GENERATOR_SET = ImageDataGenerator(
            horizontal_flip=horizon,
            vertical_flip=vertical,
            rotation_range=rotation
        )
        MSK_GENERATOR_SET = ImageDataGenerator(
            horizontal_flip=horizon,
            vertical_flip=vertical,
            rotation_range=rotation
        )
        print("here")
        IMG_GENERATOR = IMG_GENERATOR_SET.flow_from_directory(
            directory=img_dir,
            class_mode=None,
            seed=seed,
            shuffle=shuffle,
            target_size=(input_size, input_size),
            batch_size=batch
        )
        print("here 2")
        MSK_GENERATOR = MSK_GENERATOR_SET.flow_from_directory(
            directory=msk_dir,
            class_mode=None,
            seed=seed,
            shuffle=shuffle,
            target_size=(input_size, input_size),
            batch_size=batch,
            color_mode="grayscale"
        )

        DATA_SET = zip(IMG_GENERATOR, MSK_GENERATOR)
        length = IMG_GENERATOR.samples

        return DATA_SET, length