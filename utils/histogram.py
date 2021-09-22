import numpy as np
import matplotlib.pyplot as plt
from tifffile import tifffile as tifi

def histogram(image_dir1, image_dir2, bins, no_target=255):

    image1 = tifi.imread(image_dir1)
    image2 = tifi.imread(image_dir2)

    no_target_mask1 = image1 == no_target
    image1 = image1[~no_target_mask1]

    no_target_mask2 = image2 == no_target
    image2 = image2[~no_target_mask2]

    # plt.subplot(121)
    plt.hist(image1, range=(0, 1), bins=bins, color='r', edgecolor="black", alpha=0.8)
    # plt.subplot(122)
    plt.hist(image2, range=(0, 1), bins=bins, color='b', edgecolor="black", alpha=0.8)
    plt.legend(["channel att.", "naive rgbd"])

    plt.show()
    # for idx in range(bins):
    #     prob_idx = (image >= interval*idx) & (image < interval*(idx+1))
    #     frequency["idx_{0}".format(idx)] = len(image[prob_idx])

if __name__ == "__main__":

    image_dir1 = "E:\\2020_SeoulChangeDetection\\Paper\\materials\\uncalibrated_model\\CA_DRU\\misclassification\\ca_fp_10.tif"
    image_dir2 = "E:\\2020_SeoulChangeDetection\\Paper\\materials\\uncalibrated_model\\DRUd\\misclassification\\drud_fp_10.tif"
    histogram(image_dir1=image_dir1,
              image_dir2=image_dir2,
              bins=20)