import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
"""
If size of image data(ex - orthophoto) was too big, cv2 can't handle it. So, modify availability of size,
before declare cv.
"""
import numpy as np
from tifffile import tifffile as tifi
import cv2
from tqdm import tqdm

imgs_dir = "E:\\2020_SeoulChangeDetection\\Paper\\Pred_prob\\predict\\valid\\DRU_d\\yongsan"
imgs_list = os.listdir(imgs_dir)
# img_array = np.zeros((512*51, 512*21))
img_array = np.zeros((512*17, 512*33))
print(img_array.shape)
for imgs_nm in tqdm(imgs_list) :
    img_dir = os.path.join(imgs_dir, imgs_nm)
    row_idx, col_idx = int(os.path.splitext(imgs_nm)[0].split("_")[-2]), int(os.path.splitext(imgs_nm)[0].split("_")[-1])
    img = tifi.imread(img_dir)
    img_array[256*row_idx:256*row_idx+512, 256*col_idx:256*col_idx+512] = img
# img_array = img_array[:18131, 0:10165]
img_array = img_array[:8704, 0:16896]
cv2.imwrite("E:\\2020_SeoulChangeDetection\\Paper\\Pred_prob\\predict\\valid\\DRU_d\\yongsan\\yongsan_valid.tif", img_array)

# label_dir = "E:\\2020_SeoulChangeDetection\\solarP\\valid_label\\label_all"
# label_list = os.listdir(label_dir)
#
# for label_nm in tqdm(label_list) :
#     nm, ext = os.path.splitext(label_nm)
#     label_f_dir = os.path.join(label_dir, label_nm)
#     label = cv2.imread(label_f_dir, cv2.IMREAD_UNCHANGED)
#     panel_mask = label==1
#     not_panel_mask = label==0
#
#     if np.sum(panel_mask) > 0 :
#         label[panel_mask] = 5.
#     if np.sum(not_panel_mask) > 0:
#         label[not_panel_mask] = 1.
#
#     dst_dir = "E:\\2020_SeoulChangeDetection\\solarP\\valid_label\\weight_all\\" + label_nm
#     cv2.imwrite(dst_dir, label)

# anchor_scale = [8, 16, 32]
# ratio = [0.5, 1, 2] # H/W
#
# len_anchor_scale = len(anchor_scale)
# len_ratio = len(ratio)
# len_anchor_template = len_anchor_scale * len_ratio
# anchor_template = np.zeros((9, 4))
#
# for idx, scale in enumerate(anchor_scale):
#     h = scale * np.sqrt(ratio) * 16
#     w = scale / np.sqrt(ratio) * 16
#     y1 = -h/2
#     x1 = -w/2
#     y2 = h/2
#     x2 = w/2
#     print("f : ", idx*len_ratio)
#     print("b : " , (idx+1)*len_ratio)
#     print(y1, x1, y2, x2)
#     anchor_template[idx*len_ratio:(idx+1)*len_ratio, 0] = y1
#     anchor_template[idx*len_ratio:(idx+1)*len_ratio, 1] = x1
#     anchor_template[idx*len_ratio:(idx+1)*len_ratio, 2] = y2
#     anchor_template[idx*len_ratio:(idx+1)*len_ratio, 3] = x2
#
# print(anchor_template)

# dir = "E:\\2020_SeoulChangeDetection\\solarP\\train_label\\weight_all"
# label_list = os.listdir(dir)
# for label_nm in tqdm(label_list) :
#
#     label_f_dir = os.path.join(dir, label_nm)
#     label = cv2.imread(label_f_dir, cv2.IMREAD_UNCHANGED)
#     if 786432 == np.sum(label) :
#         pass
#     else :
#         print(np.sum(label))
#         print("False")


# def quantization(ground_file_dir,
#                  offground_file_dir,
#                  dst_dir=None,
#                  interval=0.1102362205,
#                  num_of_interval=254) :
#     """
#     Minimum threshold : 2m
#     Maxmium threshold : 30m
#     Under minimum threshold : 0
#     Over maximum threshold : 255
#     28m / 254 = 0.1102362205m
#     :param file_dir:
#     :return:
#     """
#     ground_file_list = os.listdir(ground_file_dir)
#     offground_file_list = os.listdir(offground_file_dir)
#     alpha = 20
#     for g_nm, fg_nm in zip(ground_file_list, offground_file_list):
#         ground_dir = os.path.join(ground_file_dir, g_nm)
#         offground_dir = os.path.join(offground_file_dir, fg_nm)
#         g_img = tifi.imread(ground_dir).astype(np.float32)
#         fg_img = tifi.imread(offground_dir).astype(np.float32)
#         ndsm = fg_img-g_img
#         ndsm[ndsm <= alpha]=0
#         for idx in range(num_of_interval):
#             if idx == 0 : before_inter = alpha
#             else : before_inter = inter
#             value = idx+1
#             inter = alpha+interval*(idx+1)
#             ndsm[(ndsm > before_inter) & (ndsm <= inter)] = value
#
#         ndsm[ndsm > inter]=255
#
#         f_dst_dir = os.path.join(dst_dir, g_nm)
#         tifi.imwrite(f_dst_dir, ndsm)
#         print("save : {0}".format(str(f_dst_dir)))
#
# folder_dir = "E:\\2020_SeoulChangeDetection\\Data\\data_generation_test\\generated_data"
# folder_target = ["junggu", "yongsan"]
# folder_target2 = ["dem_renewal_export", "quantized_ndsm"]
# data_folder = ["fg_float_clipped_256", "fg_float_clipped_256_valid", "g_float_clipped_256", "g_float_clipped_256_valid"]
#
# for target in folder_target :
#
#     dir = os.path.join(folder_dir, target)
#     img_dir = os.path.join(dir, folder_target2[0])
#     dst_dir = os.path.join(dir, folder_target2[1])
#     print("DST DIR : ", dst_dir)
#
#     for idx in range(len(data_folder)//2) :
#         fg_dir = os.path.join(img_dir, data_folder[idx])
#         g_dir = os.path.join(img_dir, data_folder[idx+2])
#
#         quantization(ground_file_dir=g_dir,
#                      offground_file_dir=fg_dir,
#                      dst_dir=dst_dir)

# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.callbacks import Callback
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Conv2D
# from tensorflow.python.keras import backend as K
# from tensorflow.keras.losses import SparseCategoricalCrossentropy
# from tensorflow.keras.metrics import MeanIoU
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import torch
# import torch.nn.functional as F

    # x = torch.tensor([[[1, 1], [1, 1], [1, 1]], [[2,2],[2,2],[2,2]], [[3,3],[3,3],[3,3]]])
    # y = torch.transpose(x, -2, -1)
    # print(x.shape, y.shape)
    # z = torch.matmul(x, y)
    # print(z.shape)
    # print(z)

# class UpdatedMeanIoU(MeanIoU) :
#     def __init__(self,
#                  y_true = None,
#                  y_pred = None,
#                  num_classes=None,
#                  name=None,
#                  dtype=None):
#         super(UpdatedMeanIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_pred = tf.math.argmax(y_pred, axis=-1)
#         return super().update_state(y_true, y_pred, sample_weight)
#
# def scheduler(epoch, lr) :
#
#     inital_lr = 0.0001
#     end_lr = 0.000001
#     decay_step = 100
#     lr = (inital_lr-end_lr)*(1-epoch/decay_step) + end_lr
#
#     return lr
#
# class MyCallback(Callback):
#
#     # def on_batch_end(self, batch, logs=None):
#     #     print("\n========== on_batch_end. ==========\n")
#     #     lr = self.model.optimizer.lr
#     #     print(lr)
#
#     def on_epoch_end(self, epoch, logs=None):
#         print("========== on_epoch_end. ==========")
#         lr = self.model.optimizer.lr
#         print(lr)
#
# def toy_model(input_shape,
#               num_classes,
#               learning_rate):
#
#     input = tf.keras.Input(shape=input_shape)
#
#     conv = Conv2D(filters=num_classes,
#                   kernel_size=3,
#                   strides=1,
#                   padding="same")(input)
#
#     model = tf.keras.Model(inputs=input,
#                            outputs=conv)
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#     model.compile(optimizer=optimizer,
#                   loss=SparseCategoricalCrossentropy(),
#                   metrics=[UpdatedMeanIoU(num_classes=num_classes)])
#     return model
#
# if __name__ == "__main__" :
#
#     img = cv2.imread("E:\\Deep_learning_dataset\\LEVIR\\train\\train_3.png", cv2.IMREAD_GRAYSCALE)
#     img[img==255] = 1
#     mask = (img != 255)
#     print(mask)
#     print(img)
#     print((mask*img).float())

    # scale = 4
    # x = torch.rand(1, 64, 64, 64*2)
    # batch_size, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)//2
    # local_x = []
    # local_y = []
    # step_h, step_w = h//scale, w//scale
    #
    # for i in range(0, scale) :
    #     for j in range(0, scale):
    #         start_x, start_y = i*step_h, j*step_w
    #         end_x, end_y = min(start_x+step_h, h), min(start_y+step_w, w)
    #         if i == scale-1 : end_x = h
    #         if j == scale-1 : end_y = w
    #         local_x += [start_x, end_x]
    #         local_y += [start_y, end_y]
    #
    # value = x
    # query = x
    # key = x
    # print("Before vqk shape : ", value.shape, query.shape, key.shape)
    #
    # value = torch.stack([value[:, :, :, :w], value[:, :, :, w:]], 4)
    # query = torch.stack([query[:, :, :, :w], query[:, :, :, w:]], 4)
    # key = torch.stack([key[:, :, :, :w], key[:, :, :, w:]], 4)
    # print("After vqk shape : ", value.shape, query.shape, key.shape)
    #
    # local_block_cnt = scale*scale*2
    # value_channels = 64
    # key_channels = 64
    #
    # #  self-attention func
    # def func(value_local, query_local, key_local):
    #     batch_size_new = value_local.size(0)
    #     h_local, w_local = value_local.size(2), value_local.size(3)
    #     # value_local shape : [batch_size_new, channel, height*width*2]
    #     value_local = value_local.contiguous().view(batch_size_new, value_channels, -1)
    #
    #     query_local = query_local.contiguous().view(batch_size_new, key_channels, -1)
    #     print("q shape : ", query_local.shape)
    #     query_local = query_local.permute(0, 2, 1)
    #     print("q shape : " , query_local.shape)
    #     key_local = key_local.contiguous().view(batch_size_new, key_channels, -1)
    #
    #     # 배치끼리만 어텐션을 수행하네? local한 어텐션을 수행 - 결국 Transformer의 MSA과 동일하다.
    #     sim_map = torch.bmm(query_local, key_local)  # batch matrix multiplication
    #     print(sim_map.shape)
    #     sim_map = (key_channels ** -.5) * sim_map # normalization
    #     sim_map = F.softmax(sim_map, dim=-1)
    #     print("sim_map shape : ", sim_map.shape)
    #     context_local = torch.bmm(value_local, sim_map.permute(0, 2, 1))
    #     # context_local = context_local.permute(0, 2, 1).contiguous()
    #     context_local = context_local.view(batch_size_new, value_channels, h_local, w_local, 2)
    #     return context_local
    #
    # v_list = [value[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in range(0, local_block_cnt, 2)]
    # v_locals = torch.cat(v_list, dim=0)
    # print("Localize v_locals : ", v_locals.shape)
    # q_list = [query[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in range(0, local_block_cnt, 2)]
    # q_locals = torch.cat(q_list, dim=0)
    # k_list = [key[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in range(0, local_block_cnt, 2)]
    # k_locals = torch.cat(k_list, dim=0)
    #
    # context_locals = func(v_locals, q_locals, k_locals)
    # print("Context locals : ", context_locals.shape)
    #
    # context_list = []
    # for i in range(0, scale):
    #     row_tmp = []
    #     for j in range(0, scale):
    #         left = batch_size * (j + i * scale)
    #         right = batch_size * (j + i * scale) + batch_size
    #         print(left, right)
    #         tmp = context_locals[left:right]
    #         print(tmp.shape)
    #         row_tmp.append(tmp)
    #     context_list.append(torch.cat(row_tmp, 3))
    #
    # context = torch.cat(context_list, 2)
    # print("Final right before : ", context.shape)
    # context = torch.cat([context[:, :, :, :, 0], context[:, :, :, :, 1]], 3)
    # print("Final : ", context.shape)


    # input_shape = (512, 512, 3)
    # num_classes = 2
    # # learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
    # #     initial_learning_rate=0.1,
    # #     decay_steps=5,
    # #     end_learning_rate=0.001,
    # #     power=1.0
    # # )
    # learning_rate = 0.01
    # BATCH_SIZE = 1
    # seed = 10
    #
    # # train
    # image_datagen = ImageDataGenerator(
    #     horizontal_flip=True,
    #     vertical_flip=True
    # )
    # mask_datagen = ImageDataGenerator(
    #     horizontal_flip=True,
    #     vertical_flip=True
    # )
    # image_generator = image_datagen.flow_from_directory(
    #    directory="E:\\deep_learning_project\\toy_set\\images",
    #    class_mode=None,
    #    seed=seed,
    #    shuffle=True,
    #    target_size=(512, 512),
    #    batch_size=BATCH_SIZE
    # )
    # mask_generator = mask_datagen.flow_from_directory(
    #    directory="E:\\deep_learning_project\\toy_set\\labels",
    #    class_mode=None,
    #    seed=seed,
    #    shuffle=True,
    #    target_size=(512, 512),
    #    color_mode="grayscale",
    #    batch_size=BATCH_SIZE
    # )
    #
    # def create_train_generator(img, label):
    #     while True:
    #         for x1, x2 in zip(img, label):
    #             yield x1, x2
    #
    # callback = [tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1), MyCallback()]
    #
    # train_gen = create_train_generator(image_generator, mask_generator)
    # model = toy_model(input_shape=input_shape,
    #                   num_classes=num_classes,
    #                   learning_rate=learning_rate)
    # model.summary()
    # model.fit(
    #     train_gen,
    #     batch_size=BATCH_SIZE,
    #     steps_per_epoch=mask_generator.samples/BATCH_SIZE,
    #     callbacks=callback,
    #     epochs=100
    # )