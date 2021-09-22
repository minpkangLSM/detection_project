import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import tifffile as tifi
from matplotlib import pyplot as plt
from skimage.measure import label
from scipy.ndimage.morphology import distance_transform_edt
import numpy as np

# ref : https://stackoverflow.com/questions/50255438/pixel-wise-loss-weight-for-image-segmentation-in-keras
def unet_weight_map(y, wc=None, w0 = 10, sigma = 5):

    """
    Generate weight maps as specified in the U-Net paper
    for boolean mask.

    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf

    Parameters
    ----------
    mask: Numpy array
        2D array of shape (image_height, image_width) representing binary mask
        of objects.
    wc: dict
        Dictionary of weight classes.
    w0: int
        Border weight parameter.
    sigma: int
        Border width parameter.

    Returns
    -------
    Numpy array
        Training weights. A 2D array of shape (image_height, image_width).
    """

    labels = label(y)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:,:,i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:,:,0]
        d2 = distances[:,:,1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
    else:
        w = np.zeros_like(y)
    if wc:
        class_weights = np.zeros_like(y)
        for k, v in wc.items():
            class_weights[y == k] = v
        w = w + class_weights
    return w

def target_weight(img, target_weight, none_target_weight=1):
    """
    weight for each class, assuming binary classes.
    :param img: semantic label data (class1 = 0, class2 = 1)
    :param target_weight: weights for target label : 1
    :param none_target_weight: weight for none target label : 0
    :return:
    """
    img_shp = np.shape(img)
    img_msk = np.zeros([img_shp[0], img_shp[1]])
    main_m = img == 1
    none_m = img == 0

    img_msk[main_m] = target_weight
    img_msk[none_m] = none_target_weight

    return img_msk

if __name__ == "__main__" :

    # class bias weights
    wc = {
        0: 1,  # background
        1: 5  # objects
    }

    dst_dir = "E:\\2020_SeoulChangeDetection\\Data\\raw_data\\swham\\annotations\\all_images_weight_normal"
    folder_dir = "E:\\2020_SeoulChangeDetection\\Data\\raw_data\\swham\\annotations\\all_images"
    image_list = os.listdir(folder_dir)

    for file in tqdm(image_list) :

        split_ext = os.path.splitext(file)
        file_dir = os.path.join(folder_dir, file)
        results_dir = os.path.join(dst_dir, file)
        img = np.array(Image.open(file_dir))

        # U-Net weight mask
        # w = unet_weight_map(img, wc, w0=10, sigma=20)
        # w = Image.fromarray(w)
        # w.save(results_dir)

        # Normal weight mask
        w = target_weight(img, target_weight=5, none_target_weight=1)
        w = Image.fromarray(w)
        w.save(results_dir)