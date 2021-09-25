from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_cd_datagen(a_path, b_path, cmap_path, target_size, batch_size, my_seed=6490):
    a_datagen = ImageDataGenerator(
        vertical_flip=True,
        horizontal_flip=True,
        zoom_range=0.1
    )
    a_generator = a_datagen.flow_from_directory(
        a_path,
        target_size=target_size,
        class_mode=None,
        seed=my_seed,
        batch_size=batch_size
    )

    b_datagen = ImageDataGenerator(
        vertical_flip=True,
        horizontal_flip=True,
        zoom_range=0.1
    )
    b_generator = b_datagen.flow_from_directory(
        b_path,
        target_size=target_size,
        class_mode=None,
        seed=my_seed,
        batch_size=batch_size
    )

    cmap_datagen = ImageDataGenerator(
        vertical_flip=True,
        horizontal_flip=True,
        zoom_range=0.1
    )
    cmap_generator = cmap_datagen.flow_from_directory(
        cmap_path,
        target_size=target_size,
        class_mode=None,
        seed=my_seed,
        batch_size=batch_size
    )

    def create_train_generator(gen_x1, gen_x2, gen_y):
        while True:
            for x1, x2, x3 in zip(gen_x1, gen_x2, gen_y):
                yield [x1, x2], x3[:, :, :, 0:1]/255

    return create_train_generator(a_generator, b_generator, cmap_generator)
