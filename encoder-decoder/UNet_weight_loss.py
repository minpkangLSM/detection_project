import os
import cv2
import numpy as np
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

img = cv2.imread("D:\\temp\\clipped\\35701_0_0.png", cv2.IMREAD_GRAYSCALE)
img_object = img>100
img_bounda = img<100
img[img_object] = 1
img[img_bounda] = 0

wc = {
    0: 1, # background
    1: 5  # objects
}
w = unet_weight_map(img, wc, w0=10, sigma=20)
w = np.array(w)
print(np.unique(w))
plt.imshow(w)
plt.show()

# if __name__ == "__main__" :
#
#     # class bias weights
#     wc = {
#         0: 1,  # background
#         1: 5  # objects
#     }
#
#     folder_dir = "D:\\temp\\clipped"
#     dst_dir = "D:\\temp\\results_temp"
#     image_list = os.listdir(folder_dir)
#     for file in image_list :
#         file_dir = os.path.join(folder_dir, file)
#         results_dir = os.path.join(dst_dir, file)
#         img = cv2.imread(file_dir, cv2.IMREAD_GRAYSCALE)
#         img_object = img > 100
#         img_bounda = img < 100
#         img[img_object] = 1
#         img[img_bounda] = 0
#
#         w = unet_weight_map(img, wc, w0=255, sigma=20)
#         cv2.imwrite(results_dir, w)

