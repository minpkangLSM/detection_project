"""
2021.05.28.
renewal tools.py
"""
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
from PIL import Image
from tifffile import tifffile as tifi
import matplotlib.pyplot as plt
from numba import jit, int64, float64
import csv
import math
import shutil
import numpy as np
import pandas as pd
import random as rd
from tqdm import tqdm

@jit(float64[:,:](float64[:,:], float64[:,:], float64[:,:,:], float64), nopython=True)
def annot2semlabel_sub(color_label_list, semantic_label_sub, label, last_index):

    semantic_label = semantic_label_sub
    for i in range(0, np.shape(label)[0]):
        for j in range(0, np.shape(label)[1]):

            # preparing for pixel to be error
            cls_ = -1
            pixel_val = label[i, j, :]
            for idx, val_list in enumerate(color_label_list):
                if (val_list[0] == pixel_val[0]) :
                    if(val_list[1] == pixel_val[1]) :
                        if (val_list[2] == pixel_val[2]) :
                            cls_ = idx
            if cls_ == -1 :
                print("========== Error : -1 ========== : No values in color_label_list. change to 4")
                cls_ = 4
            if cls_ == last_index : cls_ = cls_-1
            semantic_label[i, j] = cls_

    return semantic_label

class preprocessing :

    @staticmethod
    def annot2semlabel(frame, color_dict_pallete, ext, binary=False):

        last_index = int(len(color_dict_pallete.keys()))
        if ext == ".tif" : frame = frame.astype(np.float64)[:,:,::-1]
        else : frame = frame.astype(np.float64)

        (y, x, z) = np.shape(frame)

        if binary :
            # datatype : float64
            mask_1 = frame > 100
            mask_0 = frame < 100
            frame[mask_1] = 255
            frame[mask_0] = 0

        semantic_label_sub = np.zeros([y,x]).astype(np.float64)
        color_label_list = np.array(list(color_dict_pallete.values())).astype(np.float64)
        frame = annot2semlabel_sub(color_label_list,
                                   semantic_label_sub,
                                   frame,
                                   last_index)
        return frame

    @staticmethod
    def clip(folder_dir,
             dst_dir,
             width,
             height,
             stride,
             semantic_mode=False,
             label_dir=None,
             color_dict_pallete=None):
        """
        clip image into small size you set on "width" and "height". Marginal parts are filled with zero value.
        :parma folder_dir : directory to save clipped image(image or mask)
        :param dst_dir: folder to save result images.
        :param width : size of clipping width
        :param height : size of clipping height
        :param stride : stride to clip
        :param semantic_mode : False = just clipping, True = convert mask into semantic label wtih clipping
        :param label_dir : directory to save semantic label
        :param color_dict_pallete : dictionay type if binary type : {0 : [0, 0, 0], 1: [255, 255, 255]}
        :return:
        """
        # get image list
        img_list = os.listdir(folder_dir)
        stride = int(stride)

        for idx, img_nm in enumerate(img_list):
            name, ext = os.path.splitext(img_nm)
            img_dir = os.path.join(folder_dir, img_nm)
            # image type : tif or else
            if ext == ".tif" :
                img = tifi.imread(img_dir)
                print("TIF image. Data type gonna be converted into np.int8")
            else : img = cv2.imread(img_dir).astype(np.int8)

            shape = np.shape(img)
            wdith_size = shape[1]//width+1
            height_size = shape[0]//height+1

            for i in tqdm(range(height_size)):
                height_front = stride*i
                height_rear = height+stride*i
                for j in range(wdith_size):
                    width_front = stride*j
                    width_rear = width+stride*j

                    # case1 : RGB color image
                    if np.shape(shape)[0] == 3 :
                        frame = np.zeros([width, height, shape[-1]])
                        if height_rear > shape[0]: height_rear = shape[0]
                        if width_rear > shape[1]: width_rear = shape[1]
                        if width_rear < width_front: continue
                        if height_rear < height_front: continue
                        img_part = img[height_front: height_rear, width_front: width_rear, :]
                        frame[0:height_rear - height_front, 0:width_rear - width_front, :] = img_part
                    # case2 : Grayscale image
                    elif np.shape(shape)[0] == 2 :
                        frame = np.zeros([width, height])
                        if height_rear > shape[0]: height_rear = shape[0]
                        if width_rear > shape[1]: width_rear = shape[1]
                        if width_rear < width_front: continue
                        if height_rear < height_front: continue
                        img_part = img[height_front: height_rear, width_front: width_rear]
                        frame[0:height_rear - height_front, 0:width_rear - width_front] = img_part

                    # if semantic_mode is true
                    if semantic_mode :
                        if not label_dir : raise ValueError("Semantic mode needs directory to save semantic labels.")
                        if not color_dict_pallete : raise ValueError("Semantic mode needs pallete(dict) to convert RGB values into semantic label.")
                        semantic_frame = preprocessing.annot2semlabel(frame, color_dict_pallete, ext, binary=True).astype(np.int64)

                        if ext == "tif":
                            file_dst_dir = label_dir + "\\" + str(name) + "_{0}_{1}.tif".format(str(i), str(j))
                            if np.shape(shape)[0] == 3:
                                frame = np.flip(frame, 2)
                            tifi.imwrite(file_dst_dir, semantic_frame)
                        else:
                            file_dst_dir = label_dir + "\\" + str(name) + "_{0}_{1}.png".format(str(i), str(j))
                            cv2.imwrite(file_dst_dir, semantic_frame)

                    if ext == ".tif" :
                        file_dst_dir = dst_dir + "\\" + str(name) + "_{0}_{1}.tif".format(str(i), str(j))
                        if np.shape(shape)[0] == 3 :
                            frame = np.flip(frame, 2)
                        frame = frame.astype(np.float16)
                        tifi.imwrite(file_dst_dir, frame)
                    else :
                        file_dst_dir = dst_dir + "\\" + str(name) + "_{0}_{1}.png".format(str(i), str(j))
                        cv2.imwrite(file_dst_dir, frame)

if __name__ == "__main__":

    color_dict_pallete = {0 : [0, 0, 0], 1 : [255, 255, 255]}

    # semantic mode off
    preprocessing.clip(
        folder_dir="E:\\2020_SeoulChangeDetection\\system_test\\original",
        dst_dir="E:\\2020_SeoulChangeDetection\\system_test\\clipped",
        height=512,
        width=512,
        stride=512)

    # # semantic mode on
    # preprocessing.clip(folder_dir="E:\\2020_SeoulChangeDetection\\Data\\change_detection_set\\label\\yongsan",
    #                    dst_dir="E:\\2020_SeoulChangeDetection\\Data\\change_detection_set\\label\\yongsan_clipped_256",
    #                    height=256,
    #                    width=256,
    #                    stride=128,
    #                    semantic_mode=True,
    #                    label_dir="E:\\2020_SeoulChangeDetection\\Data\\change_detection_set\\label\\yongsan_semantic_clipped_256",
    #                    color_dict_pallete=color_dict_pallete)