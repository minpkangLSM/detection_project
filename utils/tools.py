"""
2020.05.00. ~
Kangmin Park.
This code is for pre- / post- processing dataset of Semantic Segmentation.
"""
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
"""
If size of image data(ex - orthophoto) was too big, cv2 can't handle it. So, modify availability of size,
before declare cv.
"""
import numba as nb
from numba import jit, int64, float64
import csv
import math
import cv2
import timeit
import shutil
import numpy as np
import pandas as pd
import random as rd
from tifffile import tifffile as tifi
from datetime import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from PIL import Image
from skimage.measure import label
# from sklearn.ensemble import IsolationForest
# from sklearn.neighbors import LocalOutlierFactor

"""
2020.06.03.
This class is set of functions for pre-, post-processing of U-Net data in Cultural heritage.
"""

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

@jit(float64[:,:](float64[:,:], float64[:,:], float64[:,:]), nopython=True)
def pixel_counter(p_img_matrix, t_img_matrix, temp_num_class):

    temp_conf_matrix = temp_num_class

    for i in range(0, np.shape(p_img_matrix)[0]):
        for j in range(0, np.shape(p_img_matrix)[1]):
            row = t_img_matrix[i,j]
            col = p_img_matrix[i,j]
            temp_conf_matrix[int(row), int(col)] += 1

    return temp_conf_matrix

@jit(nopython=True)
def label2class(label_file, row, col, zeros_np):
    """
    # No. class starts from 1 not 0
    :param label_file: semantic label file instance, not directory
    :param row: size of row
    :param col: size of col
    :param num_cls: class num
    :return:
    """
    x = zeros_np
    for i in range(row):
        for j in range(col):
            # No. class starts from 1 not 0
            x[int(i), int(j), int(label_file[i][j])] = 1
    return x

@jit(nopython=True)
def union(pred_inst, truth_inst):
    union = np.logical_or(pred_inst, truth_inst)
    return union

@jit(nopython=True)
def intersect(pred_inst, truth_inst):
    inter = np.logical_and(pred_inst, truth_inst)
    return inter

class preprocessing() :

    def __init__(self):
        self.value = None

    def img_chge(self,
                 folder_dir):

        # names of folders
        f_name_list = os.listdir(folder_dir)
        for folder_name in f_name_list :
            f_dir = os.path.join(folder_dir, folder_name)
            f_f_name_list = os.listdir(f_dir)

            for f_f_name in f_f_name_list :
                f_f_dir = os.path.join(f_dir, f_f_name)
                img_name_list = os.listdir(f_f_dir)
                for img_name in img_name_list :
                    re_name = folder_name + "_" + img_name
                    ori_dir = os.path.join(f_f_dir, img_name)
                    re_dir = os.path.join(f_f_dir, re_name)
                    os.rename(ori_dir, re_dir)

        print("Renaming is finished.")
        return None

    def img_format(self,
                   folder_dir,
                   format):

        img_list = os.listdir(folder_dir)
        print("Started to change format based on the input format.")
        for img_nm in img_list:
            abs_dir = os.path.join(folder_dir, img_nm)
            basename = os.path.basename(img_nm)
            filename = os.path.splitext(basename)[0] + "." + format
            dst = os.path.join(folder_dir, filename)
            os.rename(abs_dir, dst)
        print("Finish to change.")

    def img_clip(self,
                 folder_dir,
                 dst_dir,
                 width,
                 height,
                 stride):
        """
        clip image into small size you set on "width" and "height". Marginal parts are filled with zero value.
        :parma folder_dir : directory for images.
        :param dst_dir: folder to save result images.
        :param width :
        :param height :
        :param stride :
        :return:
        """
        folder_dir = folder_dir
        img_list = os.listdir(folder_dir)
        print(img_list)
        stride = int(stride)
        for idx, img_dir in enumerate(img_list) :

            print("Processing {0}th image of {1}.".format(str(idx+1), len(img_list)))
            name = os.path.splitext(img_dir)[0]

            abs_dir = os.path.join(folder_dir, img_dir)
            print(abs_dir)
            # flag : -1 = cv2.IMREAD_UNCHANGED
            _, ext = os.path.splitext(abs_dir)
            if ext == ".tif" :
                img = tifi.imread(abs_dir)
                print(img.shape)
                code = "tif"
            else :
                img = cv2.imread(abs_dir).astype(np.float64)
                code = "cv2"

            img_shp = np.shape(img)
            width_size = img_shp[1]//width+1
            height_size = img_shp[0]//height+1

            # clipping img instance and save
            for i in tqdm(range(height_size)):

                height_front = stride*i
                height_rear = height+stride*i

                for j in range(width_size):

                    width_front = stride*j
                    width_rear = width+stride*j
                    print("here : ", np.shape(img_shp))
                    print("here : ", np.shape(img_shp)[0])
                    if np.shape(img_shp)[0] == 3 :
                        print("RGB COLOR IMAGE.")
                        frame = np.zeros([width, height, img_shp[2]])

                        if height_rear > img_shp[0]: height_rear = img_shp[0]
                        if width_rear > img_shp[1]: width_rear = img_shp[1]

                        if width_rear < width_front: continue
                        if height_rear < height_front: continue

                        img_part = img[height_front: height_rear, width_front: width_rear, :]
                        frame[0:height_rear - height_front, 0:width_rear - width_front, :] = img_part

                    elif np.shape(img_shp)[0] == 2:
                        print("GRAYSCALE IMAGE.")
                        frame = np.zeros([width, height])

                        if height_rear > img_shp[0]: height_rear = img_shp[0]
                        if width_rear > img_shp[1]: width_rear = img_shp[1]

                        if width_rear < width_front: continue
                        if height_rear < height_front: continue

                        img_part = img[height_front: height_rear, width_front: width_rear]
                        frame[0:height_rear - height_front, 0:width_rear - width_front] = img_part

                    if code == "tif" :
                        file_dst_dir = dst_dir + "\\" + str(name) + "_{0}_{1}.tif".format(str(i), str(j))
                        if np.shape(img_shp)[0] == 3 :
                            frame = np.flip(frame, 2)
                        tifi.imwrite(file_dst_dir, frame)
                    elif code == "cv2" :
                        file_dst_dir = dst_dir + "\\" + str(name) + "_{0}_{1}.png".format(str(i), str(j))
                        cv2.imwrite(file_dst_dir, frame)

        print("Finished to clip and save images at {0}.".format(str(dst_dir)))

    def remove_artifact(self,
                        folder_dir,
                        dst_dir):
        """
        Remove artifact of label images.
        Pixel annotation tool을 이용하여 이진 label 데이터를 생성하면 테두리에 255값이 부여되는 버그가 있는데, 이를 제거하기 위한 코드임.
        :param folder_dir : folder location of label data
        :param dst_dir : folder to save results images.
        :return:
        """
        folder_dir = folder_dir
        img_list = os.listdir(folder_dir)

        for img_nm in img_list :
            abs_dir = os.path.join(folder_dir, img_nm)
            img = cv2.imread(abs_dir, cv2.IMREAD_GRAYSCALE)

            # remove artifact on the four corner vertex.
            if img[1, 1] < 10:
                img[0, 0] = 0
                img[0, 1] = 0
                img[1, 0] = 0
            if img[1, 510] < 10:
                img[0, 511] = 0
                img[0, 510] = 0
                img[1, 511] = 0
            if img[510, 1] < 10:
                img[510, 0] = 0
                img[511, 0] = 0
                img[511, 1] = 0
            if img[510, 510] < 10:
                img[511, 511] = 0
                img[511, 510] = 0
                img[510, 511] = 0

            # remove artifact on the four edge
            for j in list(range(1, 511)):
                if img[1, j] < 10:
                    img[0, j] = 0
                if img[510, j] < 10:
                    img[511, j] = 0

            for k in list(range(1, 511)):
                if img[k, 1] < 10:
                    img[k, 0] = 0
                if img[k, 510] < 10:
                    img[k, 511] = 0

            cv2.imwrite(dst_dir + str(img_nm), img)

    def annot2semlabels(self,
                        annot_folder_dir,
                        dst_dir,
                        color_dict_mode="manual",
                        color_dict_manual=None):
        """
        This code is for creating semantic label data for multi-classification of semantic segmentation. (2020 Landcover Semantic label)
        * What is semantic label? - refer to Evernote [Deep Learning Review]-[U-Net]
        :param annot_folder_dir:
        :param class_num: the number of classes, dtype = integer
        :param color_dict_mode : dtype-dictionary, color index key : dtype-integer(from 0), color index value : BGR order list
        option 1 : "manual", new color dictionary type.
                    color_dict_manual parameter should be declared.
        option 2 : "detail", this is based on 2020 Landcover semantic segmentation color pallete.
                    Option "main" is the main catetory of Landcover defined by Ministry of Environ.
        option 3 : "middle", this is based on arbitrary landcover semnatic segmentation color palletes.
                    related with MoE.
        option 4 : "main", same with the options.

        # CAUTION

        환경부 토지피복도 이미지 데이터에는 두 종류의 배경(흰색과 검은색)이 있다. 따라서 컬러팔레트(color_dict)에서는 두 값에 모두 고유의 키값을 부여하지만,
        흰색과 검은색 모두 "배경"이기 때문에 annot2semlabel_sub 함수에서 두 값의 키를 하나로 통일한다. 예를 들어, 세분류(detail)에서는 43(흰색)을 42 바꿔서 저장한다.
        따라서 환경부 토지피복도가 아닌 데이터를 사용할 때 반드시 이 점을 고려하여 코드를 수정해야 한다.

        :return: Grayscale semantic labels map.
        """
        # Make annotation file dir list
        annot_file_dir_list = []
        annot_file_list = os.listdir(annot_folder_dir)
        for annot_file in annot_file_list:
            annot_file_dir = os.path.join(annot_folder_dir, annot_file)
            annot_file_dir_list.append(annot_file_dir)

        if color_dict_mode == "main" :
            color_dict = { 0 : [100, 0, 255], 1 : [0, 200, 200], 2 : [100, 255, 0], 3 : [10, 100, 0], 4 : [100, 10, 80],
                           5 : [90, 90, 100], 6 : [200, 100, 10], 7 : [0, 0, 0], 8 : [255, 255, 255] }
        elif color_dict_mode == "middle" :
            color_dict = { 0 : [255, 100, 255], 1 : [70, 50, 250], 2 : [80, 100, 180], 3 : [100, 80, 190], 4 : [50, 140, 240],
                           5 : [110, 130, 250], 6 : [90, 240, 220], 7 : [50, 240, 140], 8 : [90, 230, 190], 9 : [0, 240, 240],
                           10 : [20, 170, 150], 11 : [80, 210, 30], 12 : [210, 210, 70], 13 : [230, 250, 140], 14 : [70, 130, 70],
                           15 : [50, 150, 100], 16 : [200, 80, 200], 17 : [200, 90, 100], 18 : [160, 150, 130], 19 : [130, 180, 160],
                           20 : [250, 10, 10], 21 : [200, 120, 40], 22 : [0, 0, 0], 23 : [255, 255, 255] }
        elif color_dict_mode == "detail":
            color_dict = { 0 : [194, 230, 254], 1 : [111, 193, 223], 2 : [132, 132, 192], 3 : [184, 131, 237], 4 : [164, 176, 223],
                           5 : [138, 113, 246], 6 : [254, 38, 229], 7 : [81, 50, 197], 8 : [78, 4, 252], 9 : [42, 65, 247],
                           10 : [0, 0, 115], 11 : [18, 177, 246], 12 : [0, 122, 255], 13 : [27, 88, 199], 14 : [191, 255, 255],
                           15 : [168, 230, 244], 16 : [102, 249, 247], 17 : [10, 228, 245], 18 : [115, 220, 223], 19 : [44, 177, 184],
                           20 : [18, 145, 184], 21 : [0, 100, 170], 22 : [44, 160, 51], 23 : [64, 79, 10], 24 : [51, 102, 51],
                           25 : [148, 213, 161], 26 : [90, 228, 128], 27 : [90, 176, 113], 28 : [51, 126, 96], 29 : [208, 167, 180],
                           30 : [153, 116, 153], 31 : [162, 30, 124], 32 : [236, 219, 193], 33 : [202, 197, 171], 34 : [165, 182, 171],
                           35 : [138, 90, 88], 36 : [172, 181, 123], 37 : [255, 242, 159], 38 : [255, 167, 62], 39 : [255, 109, 93],
                           40 : [255, 57, 23], 41 : [0, 0, 0], 42 : [255, 255, 255] }
        elif color_dict_mode == "ISPRS" :
            color_dict = { 0 : [0, 0, 0], 1 : [0, 0, 128], 2 : [128, 64, 128], 3 : [0, 128, 0], 4 : [0, 128, 128],
                           5 : [128, 0, 64], 6 : [192, 0, 192], 7 : [0, 64, 64] }

        elif color_dict_mode == "manual" :
            color_dict = color_dict_manual
            if color_dict == None : raise SyntaxError("if color_dict_mode is 'manaul' option, 'color_dict_manual' should be defined as dictionary type.")
        else :
            raise SyntaxError("color_dict_option includes 4 options : 'main', 'middle', 'detail' or 'manual'.")

        if color_dict_mode == "main" : last_index = 8
        elif color_dict_mode == "middle" : last_index = 23
        elif color_dict_mode =="detail" : last_index = 42
        elif color_dict_mode == "ISPRS" : last_index = 9 # None last index
        else : last_index = 2

        for idx, annot_abs_dir in enumerate(annot_file_dir_list):
            # datatyep : uint8 -> int64, [:,:,:]
            _, ext = os.path.splitext(annot_abs_dir)
            if ext == ".tif":
                label = tifi.imread(annot_abs_dir)
                label = np.array(label).astype(np.float64)
                label = label[:,:,::-1] # RGB to BGR
            else : label = cv2.imread(annot_abs_dir, cv2.IMREAD_COLOR).astype(np.float64)
            (y, x, z) = np.shape(label)  # y : height, x : width, z : depth
            # datatype : float64
            mask_1 = label > 100
            mask_0 = label < 100
            label[mask_1] = 255
            label[mask_0] = 0
            semantic_label_sub = np.zeros([y, x]).astype(np.float64)
            print("{0} / {1} processing...".format(str(idx+1), len(annot_file_dir_list)))
            # datatype : float64
            color_label_list = np.array(list(color_dict.values())).astype(np.float64)
            # numba processing
            semantic_label = annot2semlabel_sub(color_label_list, semantic_label_sub, label, last_index)
            basename = os.path.basename(annot_abs_dir)
            name = os.path.splitext(basename)[0]#[:-5]
            cv2.imwrite(str(dst_dir) + "\\" + name + ".png", semantic_label)

    def class_reducer(self,
                      image_dir,
                      label_dir,
                      image_target,
                      label_target,
                      ratio,
                      preserved) :
        img_list = os.listdir(image_dir)
        lbl_list = os.listdir(label_dir)

        for img, lbl in zip(img_list, lbl_list) :

            preserving_button = False
            img_dir = os.path.join(image_dir, img)
            lbl_dir = os.path.join(label_dir, lbl)
            img_target = os.path.join(image_target, img)
            lbl_target = os.path.join(label_target, lbl)

            lbl = cv2.imread(lbl_dir, cv2.IMREAD_GRAYSCALE)
            img_shp = lbl.shape
            denom = img_shp[0]*img_shp[1]

            mask_41 = lbl == 41
            mask_40 = lbl == 40
            mask_30 = lbl == 30
            mask_28 = lbl == 28
            mask_9 = lbl == 9

            for num in preserved :
                mask_p = lbl == num
                if np.sum(mask_p) != 0 :
                    preserving_button = True

            if not preserving_button :

                ratio_41 = np.sum(mask_41)/denom
                ratio_40 = np.sum(mask_40)/denom
                ratio_30 = np.sum(mask_30)/denom
                ratio_28 = np.sum(mask_28)/denom
                ratio_9 = np.sum(mask_9)/denom

                if ratio_41 > ratio :
                    shutil.move(img_dir, img_target)
                    shutil.move(lbl_dir, lbl_target)
                    print(lbl_dir)
                elif ratio_40 > ratio :
                    shutil.move(img_dir, img_target)
                    shutil.move(lbl_dir, lbl_target)
                    print(lbl_dir)
                elif ratio_9 > ratio :
                    shutil.move(img_dir, img_target)
                    shutil.move(lbl_dir, lbl_target)
                    print(lbl_dir)
                elif ratio_28 > ratio :
                    shutil.move(img_dir, img_target)
                    shutil.move(lbl_dir, lbl_target)
                    print(lbl_dir)
                elif ratio_30 > ratio :
                    shutil.move(img_dir, img_target)
                    shutil.move(lbl_dir, lbl_target)
                    print(lbl_dir)
        return None

    def label_change(self,
                     mask_folder_dir,
                     dst_folder):

        mask_list = os.listdir(mask_folder_dir)

        for mask in mask_list :
            mask_dir = os.path.join(mask_folder_dir, mask)
            mask_img = cv2.imread(mask_dir, cv2.IMREAD_UNCHANGED)

            for class_no in range(0, 25) :
                if (class_no == 3) or (class_no == 8) or (class_no == 19) or (class_no == 20):
                    mask_1 = mask_img == class_no
                    mask_img[mask_1] = 1
                else :
                    mask_0 = mask_img == class_no
                    mask_img[mask_0] = 0

            print(dst_folder+"\\"+mask)
            cv2.imwrite(dst_folder+"\\"+mask, mask_img)

    def quanizing(self, img_dir, quantizing_range):
        return

    class RgbSvm :

        def __init__(self):
            self.value = None

        def img2xml(self, folder_dir, result_file_dir, shuffle, sigma) :
            """
            SVM으로 어노테이션을 생성하기 학습시킬 rgb 레이블 데이터를 생성하는 코드입니다.
            :param folder_dir: 샘플 이미지 별로 레이블링 된 폴더의 위치가 입력되어야 합니다.
            :param dst_dir: 생성된 rgb label의 xml을 저장할 위치를 입력합니다.
            :param shuffle: 파일 순서를 random으로 Shuffle : True or False
            :param sigma: : Noise 제거를 위한 Gaussian filer의 sigma값을 설정합니다.
            :return:
            """

            # make image folder directory shuffle or not
            label_folder_list = os.listdir(folder_dir)
            label_folder_dir_list = []
            images_dir_dict = {}

            for label_folder in label_folder_list :
                label_folder_dir = os.path.join(folder_dir, label_folder)
                sample_img_list = os.listdir(label_folder_dir)

                img_dir_list = []
                if shuffle == True : rd.shuffle(sample_img_list)

                for img_name in sample_img_list :
                    img_dir = os.path.join(label_folder_dir, img_name)
                    img_dir_list.append(img_dir)

                images_dir_dict[str(label_folder)] = img_dir_list

            if math.floor((6*sigma)%2) == 0 : ksize = 6*sigma+1
            else : ksize = math.floor(6*sigma)
            kernel = (ksize, ksize)
            data_set = []

            for set_key, set_img_dir_list in images_dir_dict.items() :
                for img_dir in set_img_dir_list :

                    img_bgr = cv2.imread(str(img_dir)).astype(np.float64)
                    img_shp = np.shape(img_bgr)
                    blurred = cv2.blur(img_bgr, kernel)
                    blurred = np.reshape(blurred, [-1,3])

                    for i in range(np.shape(blurred)[0]):
                        # if outlier[i] == 1:
                        if True:
                            pxl = blurred[i]
                            bgr_set = {"b": pxl[0], "g": pxl[1], "r": pxl[2], "label": int(set_key)}
                            data_set.append(bgr_set)

            if shuffle == True: rd.shuffle(data_set)
            csv_columns = ["b", "g", "r", "label"]
            with open(result_file_dir, 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
                writer.writeheader()
                for data in data_set:
                    writer.writerow(data)

        def annotation(self, test_dir, train_file_dir, dst_dir, degree, coef0, C):

            # Make abs image file directory (test image)
            img_list = os.listdir(test_dir)
            img_dir_list = []
            for img_name in img_list :
                img_dir = os.path.join(test_dir, img_name)
                img_dir_list.append(img_dir)

            l_dataset = np.array(pd.read_csv(train_file_dir))
            data_x = l_dataset[:,0:3]
            data_y = l_dataset[:,-1]

            polynominal_svm_clf = Pipeline([
                ("scaler", StandardScaler()),
                ("svm_clf", SVC(kernel="poly", degree=degree, coef0=coef0, C=C))
            ])
            print("Started annotation {0} files, using SVM.".format(str(len(img_dir_list))))
            count = 0
            for img_dir in img_dir_list :
                count += 1
                img_bgr = cv2.imread(img_dir).astype(np.float64)
                img_shp = np.shape(img_bgr)
                img_rslt = np.zeros([img_shp[0], img_shp[1]])
                polynominal_svm_clf.fit(data_x, data_y)

                for i in range(img_shp[0]) :
                    for j in range(img_shp[1]) :
                        apart = np.reshape(img_bgr[i,j], [1,-1])
                        predict = polynominal_svm_clf.predict(apart)
                        if predict[0] == 0 : img_rslt[i,j] = 255;
                file_nm = os.path.basename(img_dir)
                print("Processed {0} / {1} : {2}%".format(str(count), str(len(img_dir_list)), str(count/len(img_dir_list))))
                dir = dst_dir + "\\" + str(file_nm)
                cv2.imwrite(dir, img_rslt)

class postprocessing :

    def __init__(self):
        self.value = None

    def merge_results(self, folder_dir, width, height, result_name, result_dst, image_size = 512):
        """
        Code for merging label results of U-Net into an image.
        U-Net 레이블 결과가 저장되어있는 폴더에는 U-net에 필요한 .txt와 .npy 파일이 있음에 유의한다.(U-Net_Git2의 result 폴더)
        :param folder_dir : result label file directory
        :param width : result image width size
        :param height : result image height size
        :param result_name : name
        :param result_dst : destination directory of result image
        :return: None
        """
        # Extracting total number of images, row and col of last image(copped)
        img_list = os.listdir(folder_dir)
        f_row = 0
        f_col = 0
        for idx, img_nm in enumerate(img_list):
            split = os.path.splitext(img_nm)[0]
            _, row, col = split.split("_")
            if idx == 0:
                f_row = int(row)
                f_col = int(col)
            else:
                if f_row < int(row): f_row = int(row)
                if f_col < int(col): f_col = int(col)

        # Merging copped images
        result_array = np.zeros([(f_row + 1) * image_size, (f_col + 1) * image_size])
        for result_img in img_list:
            img_dir = folder_dir + "\\" + result_img
            result_img = os.path.splitext(result_img)[0]
            img_nm_split = result_img.split("_")
            print(img_nm_split)
            row = int(img_nm_split[1])
            col = int(img_nm_split[2])
            img = tifi.imread(img_dir)
            result_array[row * image_size:row * image_size + image_size,
            col * image_size:col * image_size + image_size] = img

        result_array = result_array[0:int(height), 0:int(width)]
        dir = result_dst + "\\" + result_name
        tifi.imwrite(dir, result_array)
        #cv2.imwrite(dir, result_array)
        print("Complete merging task.")

    def count_pixel(self, folder_dir, class_num):

        # pixel count list
        pixel_list = np.zeros(class_num, dtype=np.float64)

        # import semantic labels
        img_list = os.listdir(folder_dir)

        for img_file in tqdm(img_list) :
            img_abs_dir = os.path.join(folder_dir, img_file)
            _, ext = os.path.splitext(img_abs_dir)
            if ext == ".tif":
                img = tifi.imread(img_abs_dir)
            else:
                img = cv2.imread(img_abs_dir, cv2.IMREAD_UNCHANGED)
            for cls in range(0, class_num) :
                mask = img==cls
                pixel_list[cls] += np.sum(mask)

        with open(folder_dir + "\\" + "num_of_pixel_in_datasets.csv", 'w', newline='') as f :
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow(pixel_list)

    def confusion_matrix(self, t_folder, p_folder, num_class):

        conf_matrix = np.zeros([num_class, num_class])
        conf_matrix_rate = np.zeros([num_class, num_class])
        p_file_list = os.listdir(p_folder)
        t_file_list = os.listdir(t_folder)
        print("Total images : {0}".format(len(p_file_list)))
        count = 0

        for p_file, t_file in zip(p_file_list, t_file_list) :
            count+=1
            print(p_file)
            print(t_file)
            print("{0} / {1} processing.".format(count, len(p_file_list)))
            temp_num_class = np.zeros([int(num_class), int(num_class)]).astype(np.float64)

            p_abs_dir = os.path.join(p_folder, p_file)
            t_abs_dir = os.path.join(t_folder, t_file)

            p_img = cv2.imread(p_abs_dir, cv2.IMREAD_UNCHANGED).astype(np.float64)
            t_img = cv2.imread(t_abs_dir, cv2.IMREAD_UNCHANGED).astype(np.float64)
            print(p_img)
            print(t_img)
            temp_num_class = pixel_counter(p_img, t_img, temp_num_class)
            conf_matrix += temp_num_class

        #sum = np.sum(conf_matrix, 0)
        for i in range(num_class):
            conf_matrix_rate[:,i] = conf_matrix[:,i]

        plt.matshow(conf_matrix_rate, cmap=plt.cm.gray_r)
        plt.colorbar()
        plt.show()
        np.savetxt("confusion_matrix_rate.txt", conf_matrix_rate)

    def binary_analysis(self, ref_dir, pre_dir, result_map_nm=None):

        img_ref = np.array([cv2.imread(ref_dir, cv2.IMREAD_GRAYSCALE)]).transpose(1,2,0)
        img_pre = np.array([cv2.imread(pre_dir, cv2.IMREAD_GRAYSCALE)]).transpose(1,2,0)
        img_shape = img_ref.shape

        pre_true_mask = img_pre > 100
        pre_false_mask = img_pre < 100
        ref_true_mask = img_ref > 100
        ref_false_mask = img_ref < 100

        img_ref[ref_true_mask] = 1
        img_ref[ref_false_mask] = 0
        img_pre[pre_true_mask] = 1
        img_pre[pre_false_mask] = 0

        TP = np.sum((img_ref == 1) & (img_pre == 1))
        TN = np.sum((img_ref == 0) & (img_pre == 0))
        FP = np.sum((img_ref == 0) & (img_pre == 1))
        FN = np.sum((img_ref == 1) & (img_pre == 0))
        TOTAL = TP + TN + FP + FN
        print("TOTAL : ", TOTAL)
        print("REF true : ", np.sum(img_ref), np.sum(img_ref)/ TOTAL)
        print("REF false : ", TOTAL - np.sum(img_ref), (TOTAL - np.sum(img_ref))/TOTAL)
        print("True positive / REF POSITIVE: ", TP / np.sum(img_ref), TP)
        print("True negative / REF NEGATIVE: ", TN / (TOTAL - np.sum(img_ref)), TN)
        print("True positive / TOTAL: ", TP/TOTAL, TP)
        print("True negative / TOTAL: ", TN/TOTAL, TN)
        print("False positive / TOTAL: ", FP/TOTAL, FP)
        print("False negative / TOTAL: ", FN/TOTAL, FN)
        print("iou : ", TP/(TP+FP+FN))
        print("precision : ", TP/(TP+FP))
        print("recall : ", TP/(TP+FN))

        if result_map_nm :

            img_zero = np.zeros([img_shape[0], img_shape[1], 3])

            for i in tqdm(range(img_pre.shape[0])) :
                for j in range(img_pre.shape[1]) :
                    if (img_ref[i,j] == 1) and (img_pre[i,j] == 1) :
                        # True Positive
                        img_zero[i, j, :] = [255, 255, 255]
                    elif (img_ref[i,j] == 0) and (img_pre[i,j] == 0) :
                        # True Negative
                        img_zero[i, j, :] = [0, 0, 0]
                    elif (img_ref[i,j] == 1) and (img_pre[i,j] == 0) :
                        # Blue : False Negative
                        img_zero[i, j, :] = [255, 0, 0]
                    elif (img_ref[i,j] == 0) and (img_pre[i,j] == 1) :
                        # Green : False Positive
                        img_zero[i, j, :] = [0, 255, 0]

            cv2.imwrite(result_map_nm, img_zero)

    def object_count(self, img_dir, threshold=0.5):
        """

        :param img_dir: predict -> clipped by ground truth -> 255 : TP, 0 : FN, 125 : None building area
        :param threshold: for predciting whether building or not
        :return:
        """
        img = tifi.imread(img_dir)
        img_re = img[:,:,0]
        img_init = np.copy(img)[:,:,0]
        results = np.zeros_like(img_init)

        mask_255 = img_re==255
        mask_125 = img_re==125
        mask_0 = img_re==0

        # img rescaling
        # 1 : building / 0 : none building area
        img_re[mask_255] = 1
        img_re[mask_0] = 1
        img_re[mask_125] = 0

        # img rescaling
        img_init[mask_255] = 1
        img_init[mask_0] = 0

        labels = label(img_re, neighbors=4)
        num = int(len(np.unique(labels)))
        print(num)
        count = 0

        for i in tqdm(range(1, 10)) :
            mask = labels==i
            ground = np.sum(img_re[mask])
            pred = np.sum(img_init[np.where(mask)])
            if pred/ground > threshold :
                count += 1
                results[np.where(mask)] = 1
            else :
                results[np.where(mask)] = 2

        print("Total : ", num-1)
        print("Pred count : ", count)
        print("Total percentage : ", count*100/(num-1))
        print(results)
        print(np.unique(results))
        tifi.imwrite("E:\\2020_SeoulChangeDetection\\Data\\data_generation_test\\image3.tif", results)

    def false_positive(self, ground_dir, predict_dir):

        ground_truth = tifi.imread(ground_dir)
        print(ground_truth.shape)
        predict = cv2.imread(predict_dir)
        print(predict.shape)
        results = np.zeros_like(predict)
        print(results.shape)

        g = ground_truth[:,:,0]
        p = predict[:,:,0]

        mask_1 = g==255
        mask_2 = p==255
        g[mask_1] = 1
        p[mask_2] = 1
        g[~mask_1] = 0
        p[~mask_2] = 0

        for i in tqdm(range(predict.shape[0])):
            for j in range(predict.shape[1]):
                if g[i,j] == 0 and p[i,j] == 1 :
                    results[i,j,:] = [255, 255, 255]

        tifi.imwrite("E:\\2020_SeoulChangeDetection\\Data\\data_generation_test\\generated_data\\myeonmok\\false_positive.tif", results)


if __name__ == "__main__" :

    # g_dir = "E:\\2020_SeoulChangeDetection\\Data\\data_generation_test\\generated_data\\myeonmok\\clipped\\results_clipped\\groundt_forobject.tif"
    # p_dir = "E:\\2020_SeoulChangeDetection\\Model\\results\\test_myeonmok.jpg"
    # inst = postprocessing()
    # inst.false_positive(ground_dir=g_dir, predict_dir=p_dir)

    # img_dir = "E:\\2020_SeoulChangeDetection\\Data\\data_generation_test\\generated_data\\myeonmok\\clipped\\results_clipped\\clipped3.tif"
    # inst = postprocessing()
    # inst.object_count(img_dir=img_dir)

    # img = cv2.imread("aachen_000000_000019_gtFine_labelIds.PNG", cv2.IMREAD_UNCHANGED)
    # print(img)
    # # img_rename
    # folder_dir = "E:\\Data_list\\Deep_learning_dataset\\ISPRS_bechmark\\uavid_v1.5_official_release_image\\uavid_v1.5_official_release_image\\uavid_test"
    # instance = preprocessing()
    # instance.img_chge(folder_dir=folder_dir)

    # img_dir = "E:\\2020_Landcover\\Data\\Landcover_mask\\changwon\\changwon_updated_mask(we1)_middle_semlabel"
    # def temp(img_dir):
    #     img_list = os.listdir(img_dir)
    #     for idx, img in enumerate(img_list) :
    #         img_abs = os.path.join(img_dir, img)
    #         print(img_abs)
    #         renm_temp = os.path.splitext(img)[0]
    #         renm = renm_temp[:-5]+".png"
    #         print(renm)
    #         img_abs2 = os.path.join(img_dir, renm)
    #         print("{0}/{1}".format(idx+1, len(img_list)))
    #         os.rename(img_abs, img_abs2)
    # temp(img_dir)

    # # print("Testing part")
    # # Testing for format changing
    # folder_dir = "E:\\2020_SeoulChangeDetection\\Data\\data_generation_test\\generated_data\\sewoon\\clipped\\dem_jpg"
    # instance = preprocessing()
    # instance.img_format(folder_dir, "jpg")

    # # Testing for image clipping of preprocessing
    # folder_dir = "E:\\2020_SeoulChangeDetection\\Paper\\out_of_distribution\\original"
    # width = 512
    # height = 512
    # dst_dir = "E:\\2020_SeoulChangeDetection\\Paper\\out_of_distribution\\clipped"
    # instance = preprocessing()
    # instance.img_clip(folder_dir=folder_dir, dst_dir=dst_dir, width=width, height=height, stride=512)

    # Testing for merging result images of postprocessing
    folder_dir = "E:\\2020_SeoulChangeDetection\\Paper\\Pred_prob\\ensemble\\DRU_DA\\m14\\dist5"
    # height = 5745
    # width = 4254
    # height = 6380
    # width = 5379
    # height = 8866
    # width = 4890
    # height = 7866
    # width = 8430
    height = 4597
    width = 3134
    result_name = "dru_da_m14_dist5.tif"
    result_dst = "E:\\2020_SeoulChangeDetection\\Paper\\Pred_prob\\ensemble\\DRU_DA\\merge"
    instance = postprocessing()
    instance.merge_results(folder_dir = folder_dir, width = width, height = height, result_name = result_name, result_dst = result_dst)

    # # Testing for creating svm rgb dataset and annotation
    # folder_dir = "E:\\Data_list\\master_paper\\cultural_heritage\\cultural_data\\SVM_preprocessing_data\\DL_dset\\moss"
    # instance = preprocessing.RgbSvm()
    # instance.img2xml(folder_dir = folder_dir, result_file_dir = "E:\\Data_list\\master_paper\\cultural_heritage\\cultural_data\\SVM_preprocessing_data\\svm_train_dset.csv", shuffle = True, sigma=2.5)
    # instance.annotation(test_dir = "E:\\Data_list\\master_paper\\cultural_heritage\\cultural_data\\SVM_preprocessing_data\\test_set",
    #                     train_file_dir = "E:\\Data_list\master_paper\\cultural_heritage\\cultural_data\\SVM_preprocessing_data\\svm_train_dset.csv", dst_dir = "E:\\Data_list\\master_paper\\cultural_heritage\\cultural_data\\SVM_preprocessing_data\\test_rlst",
    #                     degree = 5, coef0 = 1, C = 10)

    # # Testing for creating semantic labels
    # instance = preprocessing()
    # folder_dir = "E:\\Deep_learning_dataset\\LEVIR\\val\\label"
    # dst_folder = "E:\\Deep_learning_dataset\\LEVIR\\val\\semantic_label"
    # # For binary
    # color_dict_manual = { 0 : [0, 0, 0], 1 : [255, 255, 255]}
    # # For vegetation
    # # color_dict_manual = { 0 : [0, 0, 0], 1 : [128, 64, 128], 2 : [0, 76, 130], 3 : [0, 102, 0], 4 : [87, 103, 112],
    # #                       5 : [168, 42, 28], 6 : [30, 41, 48], 7 : [89, 50, 0], 8 : [35, 142, 107], 9 : [70, 70, 70],
    # #                       10 : [156, 102, 102], 11 : [12, 228, 254], 12 : [12, 148, 254], 13 : [153, 153, 190], 14 : [153, 153, 153],
    # #                       15 : [96, 22, 255], 16 : [0, 51, 102], 17 : [150, 143, 9], 18 : [32, 11, 119], 19 : [0, 51, 51],
    # #                       20 : [190, 250, 190], 22 : [146, 150, 112], 23 : [115, 135, 2], 24 : [0, 0, 255]}
    # instance.annot2semlabels(annot_folder_dir=folder_dir, dst_dir=dst_folder, color_dict_mode="manual", color_dict_manual=color_dict_manual)

    # # Tester : counting training dataset pixel
    # folder_dir = "E:\\2020_SeoulChangeDetection\\Data\\vegetation_detection\\semantic_drone_dataset_semantics_v1.1\\semantic_drone_dataset\\training_set\\gt\\semantic\\semantic_label_clipped"
    # instance = postprocessing()
    # instance.count_pixel(folder_dir, 2)

    # # Testing for confusion matrix
    # t_semlabel = "E:\\Data_list\\2020_Landcover\\Data\\Landcover_mask\\changwon\\changwon_updated_mask(ea1)_details_semlabel"
    # p_semlabel = "E:\\Data_list\\2020_Landcover\\test_result_sl3"
    # instance = postprocessing()
    # instance.confusion_matrix(t_semlabel, p_semlabel, 42)

    # # Testing for mIoU
    # p_semlabel = "test_results_semlabel"
    # t_semlabel = "test_results_truth"

    # instance = preprocessing()
    # instance.mIoU(p_semlabel, t_semlabel, 1024, 1024, 42)

    # # # label change test
    # folder_dir = "E:\\2020_SeoulChangeDetection\\Data\\vegetation_detection\\semantic_drone_dataset_semantics_v1.1\\semantic_drone_dataset\\training_set\\gt\\semantic\\label_images_semantic"
    # dst_dir = "E:\\2020_SeoulChangeDetection\\Data\\vegetation_detection\\semantic_drone_dataset_semantics_v1.1\\semantic_drone_dataset\\training_set\\gt\\semantic\\label_images_semantic_changed"
    # inst = preprocessing()
    # inst.label_change(folder_dir, dst_dir)

    # # Tester : pixel reducer
    # label_folder_dir =
    # image_folder_dir =
    # target_label = {}
    # none_target_label = {}
    # recent_target_num = {}
    # target_limit = {}

    # img = Image.open("E:\\2020_Landcover\\Data\\Landcover_mask\\37612_dataset\\37612_updated_mask_clip(bt1)_middle_semlabel\\37612_bt1_0_50.png")
    # arr = np.array(img)
    # np.savetxt("image.txt", arr)

    # # class reducer
    # img_dir = "E:\\2020_RS_journal\\Data\\dataset\\stride=256\\img_256"
    # lbl_dir = "E:\\2020_RS_journal\\Data\\dataset\\stride=256\\lbl_256"
    # img_target = "E:\\2020_RS_journal\\Data\\dataset\\stride=256\\img_256_target"
    # lbl_target = "E:\\2020_RS_journal\\Data\\dataset\\stride=256\\lbl_256_target"
    # preserved = [1, 5, 6, 8, 10, 11, 12, 13, 15, 16,
    #              20, 21, 23, 27, 31, 32, 33, 34, 35, 36, 38]
    # insta = preprocessing()
    # insta.class_reducer(img_dir, lbl_dir, img_target, lbl_target, 0.5, preserved)

    # img = cv2.imread("E:\\2020_SeoulChangeDetection\\Data\\test_images\\test_msk\\z\\image\\35701.png", cv2.IMREAD_GRAYSCALE)
    # img_shp = img.shape
    # row_index = int(img_shp[0]/2)
    # col_index = int(img_shp[1]/2)

    # img_0_0 = img[0:row_index, 0:col_index]
    # img_0_1 = img[0:row_index, col_index:img_shp[1]]
    # img_1_0 = img[row_index:img_shp[0], 0:col_index]
    # img_1_1 = img[row_index:img_shp[0], col_index:img_shp[1]]

    # cv2.imwrite("E:\\2020_SeoulChangeDetection\\Data\\test_images\\test_msk\\z\\image\\35701_0_0.png", img_0_0)
    # cv2.imwrite("E:\\2020_SeoulChangeDetection\\Data\\test_images\\test_msk\\z\\image\\35701_0_1.png", img_0_1)
    # cv2.imwrite("E:\\2020_SeoulChangeDetection\\Data\\test_images\\test_msk\\z\\image\\35701_1_0.png", img_1_0)
    # cv2.imwrite("E:\\2020_SeoulChangeDetection\\Data\\test_images\\test_msk\\z\\image\\35701_1_1.png", img_1_1)

    # # binary Analysis
    # ref_dir = "E:\\2020_SeoulChangeDetection\\Paper\\Pred_results(Naive)\\Groundtruth\\dist_5.jpg"
    # pred_dir = "E:\\2020_SeoulChangeDetection\\Paper\\Pred_results(Naive)\\DA_pred\\dist5_pred_merge\\dist5_DA.jpg"
    # dst_dir = "E:\\2020_SeoulChangeDetection\\Paper\Pred_results(Naive)\\DA_pred\\dist5_pred_merge\\dist5_RGBD_analysis.jpg"
    # instance = postprocessing()
    # instance.binary_analysis(ref_dir=ref_dir, pre_dir=pred_dir, result_map_nm=dst_dir)

    # img_dir = "E:\\2020_SeoulChangeDetection\\Data\\Seoul_images\predict_results\\35701_e96_unet_loss_weight_dru\\35701_pred_weight_DRU_merged\\35701_pred_weight_DRU_merged.jpg"
    # img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    # ero = cv2.erode(img, kernel=kernel, iterations=2)
    # dil = cv2.dilate(img, kernel=kernel, iterations=1)
    # cv2.imwrite("E:\\2020_SeoulChangeDetection\\Data\\Seoul_images\predict_results\\35701_e96_unet_loss_weight_dru\\35701_pred_weight_DRU_merged\\test.jpg", ero)

    # img = cv2.imread("E:\\2020_SeoulChangeDetection\\Data\\vegetation_detection\\semantic_drone_dataset_semantics_v1.1\\semantic_drone_dataset\\training_set\\gt\\semantic\\label_images\\000.png")
    # print(img[20,10,:])
    # img = np.reshape(img, newshape=[img.shape[0]*img.shape[1], img.shape[2]])
    # print(img)
    # print(img[200,:])
    # print(np.unique(img))

    # img = tifi.imread("E:\\2020_SeoulChangeDetection\\Data\\data_generation_test\\generated_data\\sewoon\\clipped\\dem\\sewoon_19_9_1.tif")
    # print(np.unique(img))

    # GT_img = tifi.imread("E:\\2020_SeoulChangeDetection\\Data\\Solar_temp\\sangam11GT.tif")
    # PR_img = tifi.imread("E:\\2020_SeoulChangeDetection\\Data\\Solar_temp\\sangam11pred.tif")
    # GT_img[GT_img == 1] = 255
    #
    # cv2.imwrite("E:\\2020_SeoulChangeDetection\\Data\\Solar_temp\\sangam1_gt.jpg", GT_img)
    # cv2.imwrite("E:\\2020_SeoulChangeDetection\\Data\\Solar_temp\\sangam1_pr.jpg", PR_img)
