"""
2021.06.22.
This code is for generating "fake" ndsm data.
The fake ndsm data is used for pre-training DeepResUNet rgbd version.
"""
import cv2
from tifffile import tifffile as tifi
import numpy as np
from tqdm import tqdm

def fake_ndsm(dst_dir,
              num_of_file=172750,
              shape=512,
              sigma=1,
              alpha=0,):

    for idx in tqdm(range(num_of_file)):
        fake_ndsm = alpha + np.random.random((shape, shape)) * sigma
        tifi.imsave(dst_dir+"/"+"{0}.tiff".format(idx+1), fake_ndsm)

if __name__ == "__main__" :

    dst_dir = "E:\\2020_SeoulChangeDetection\\Model\\temp\\fake_ground"
    fake_ndsm(dst_dir=dst_dir)