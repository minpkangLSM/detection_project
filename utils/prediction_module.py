import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
from tifffile import tifffile as tifi
import math
import numpy as np
#from SpatialAttention_DRU_rgb import SA_DeepResUNet
#from ChannelAttention_DRU_rgb import CA_DeepResUNet
#from DualAttention_DRU_rgb import DA_DeepResUNet
#from DeepResUNet import DeepResUNet
from DeepResUNet_rgbd_trans import DeepResUNet
#from DualAttention_DRU_rgb_platt import DA_DeepResUNet

sub_size = 512
classify = False

img_folder_dir = "./prediction_prob/raw_data/pred"
img_list = os.listdir(img_folder_dir)
fgrd_folder_dir = "./prediction_prob/fgrd_data/pred"
fgrd_list = os.listdir(fgrd_folder_dir)
grd_folder_dir = "./prediction_prob/grd_data/pred"
grd_list = os.listdir(grd_folder_dir)

img_list.sort()
fgrd_list.sort()
grd_list.sort()

DeepResUNet = DeepResUNet(input_shape=(sub_size, sub_size, 3),
                          offground_shape=(sub_size, sub_size, 1),
                          ground_shape=(sub_size, sub_size, 1),
                          weight_decay=None,
                          lr=0.01,
                          num_classes=2)
                             # ground_shape=(sub_size, sub_size, 1),
                             # offground_shape=(sub_size, sub_size, 1),
                             # weight_decay=None)

model = DeepResUNet.build_net()

## Uncalibrated
# model.load_weights("./DRU_SA_rgb_callbacks/finetune_weights/DRU_SA_rgb_49.hdf5") # SA
# model.load_weights("./DRU_CA_rgb_callbacks/finetune_weights/DRU_CA_rgb_49.hdf5") # CA
# model.load_weights("./DRU_DA_rgb_callbacks/finetune_weights/DRU_DA_rgb_44.hdf5") # DA
# model.load_weights("./renewal_dataset_0324_weight(Pretrained_finetune)/DRU_20.hdf5") # DRU
model.load_weights("./DRU_rgbd_weights_float(finetune_b_on_rgb)/DRU_48.hdf5") # DRU_d

## Calibrated
# model.load_weights("./DRU_DA_rgb_callbacks/calibrated_temp_weights_main/DRU_DA_rgb_calib_temp_50.hdf5")

for img_nm, fgrd_nm, grd_nm in zip(img_list, fgrd_list, grd_list) :
    img_dir = os.path.join(img_folder_dir, img_nm)
    grd_dir = os.path.join(grd_folder_dir, grd_nm)
    fgrd_dir = os.path.join(fgrd_folder_dir, fgrd_nm)

    _, ext = os.path.splitext(img_dir)
    if ext == ".tif":
        # img = tifi.imread(img_dir)[:,:,0:3]
        img = tifi.imread(img_dir)
    else :
        img = cv2.imread(img_dir)
    fgrd = tifi.imread(fgrd_dir)
    grd = tifi.imread(grd_dir)

    image_height, image_width, c = img.shape
    frame_height, frame_width = math.ceil(image_height/sub_size)*sub_size, math.ceil(image_width/sub_size)*sub_size

    image_frame = np.zeros((frame_height, frame_width, c))
    fgrd_frame = np.zeros((frame_height, frame_width))
    grd_frame = np.zeros((frame_height, frame_width))

    image_frame[0:image_height, 0:image_width, :] = img
    fgrd_frame[0:image_height, 0:image_width] = fgrd
    grd_frame[0:image_height, 0:image_width] = grd

    result_frame = np.zeros((frame_height, frame_width))

    for row_idx in range(int(frame_height/sub_size)):
        for col_idx in range(int(frame_width/sub_size)):
            sub_image = image_frame[sub_size*row_idx:sub_size*row_idx+sub_size,
                                    sub_size*col_idx:sub_size*col_idx+sub_size,
                                    :]
            sub_fgrd = fgrd_frame[sub_size*row_idx:sub_size*row_idx+sub_size,
                                 sub_size*col_idx:sub_size*col_idx+sub_size]
            sub_grd = fgrd_frame[sub_size * row_idx:sub_size * row_idx + sub_size,
                                sub_size * col_idx:sub_size * col_idx + sub_size]

            _sum_img = np.sum(sub_image, axis=-1)
            _sub_idx = _sum_img == 0

            # sub_image = np.expand_dims(sub_image, axis=0)
            # sub_fgrd = np.expand_dims(sub_fgrd, axis=0)
            # sub_grd = np.expand_dims(sub_grd, axis=0)

            input_data = (sub_image, sub_fgrd, sub_grd)
            # input_data = sub_image
            pred_img = model.predict(input_data,
                                     batch_size=1,
                                     verbose=1)
            pred_img = pred_img.reshape(512, 512, 2)
            result_frame[sub_size*row_idx:sub_size*row_idx+sub_size,
                         sub_size*col_idx:sub_size*col_idx+sub_size] = pred_img[:, :, 1]
            # result_frame[sub_size*row_idx:sub_size*row_idx+sub_size,
            #              sub_size*col_idx:sub_size*col_idx+sub_size][_sub_idx] = -1 # background index
    result_frame = result_frame[:image_height, :image_width]
    # if classify :
    #     classify_frame = result_frame
    #     mask1 = result_frame>=0.5
    #     mask0 = result_frame<0.5
    #     classify_frame[mask1] = 1
    #     classify_frame[mask0] = 0
    #     tifi.imwrite("./prediction_prob/valid_area/DRU_{0}_pred_results_classify.jpg".format(os.path.splitext(img_nm)[0]), classify_frame)
    tifi.imwrite("./prediction_prob/DRU_rgbd_{0}_pred_results.tif".format(os.path.splitext(img_nm)[0]), result_frame)