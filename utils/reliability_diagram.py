import os
import cv2
import pickle
from tifffile import tifffile as tifi
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# def build_diagram(pred_img, label, bins=15):
#
#     acc_list = np.zeros((bins))
#     con_list = np.zeros((bins))
#     num_list = np.zeros((bins))
#     mask = {}
#     pred = {}
#     lbl = {}
#
#     for idx in range(bins):
#         sub_mask = (pred_img >= 0.0667*idx) & (pred_img < 0.0667*(1+idx))
#         mask["bin_{0}_mask".format(idx+1)] = sub_mask
#
#     for idx, sub_mask_id in enumerate(mask.keys()) :
#         sub_pred = pred_img[mask[sub_mask_id]]
#         pred["pred_{0}_mask".format(idx+1)] = sub_pred
#
#     for idx, sub_mask_id in enumerate(mask.keys()) :
#         sub_lbl = label[mask[sub_mask_id]]
#         lbl["lbl_{0}_mask".format(idx+1)] = sub_lbl
#
#     for idx in range(bins):
#         pred_sub = pred["pred_{0}_mask".format(idx+1)]
#         lbls_sub = lbl["lbl_{0}_mask".format(idx+1)]
#         acc_list[idx] = np.sum(lbls_sub)/(len(pred_sub)+0.0001)
#         con_list[idx] = np.sum(pred_sub)/(len(pred_sub)+0.0001)
#         num_list[idx] = len(pred_sub)
#
#     return acc_list, con_list, num_list
#
#
# label_folders_dir = "E:\\2020_SeoulChangeDetection\\Paper\\Pred_prob\\label\\pred"
# label_list = os.listdir(label_folders_dir)
#
# pred_folders_dir = "E:\\2020_SeoulChangeDetection\\Paper\\Pred_prob\\ensemble\\DRU_DA\\ensemble(m1~m10)"
# pred_list = os.listdir(pred_folders_dir)
#
# data = {
#     "acc" : 0,
#     "conf" : 0,
#     "nums" : 0
# }
# count = 0
# for label_nm, pred_nm in zip(label_list, pred_list):
#     count+=1
#     label_dir = os.path.join(label_folders_dir, label_nm)
#     pred_dir = os.path.join(pred_folders_dir, pred_nm)
#
#     pred_img = tifi.imread(pred_dir)
#     label_img = cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE)
#     mask1 = label_img >= 100
#     mask0 = label_img < 100
#     label_img[mask1] = 1
#     label_img[mask0] = 0
#
#     Acc, Conf, Nums = build_diagram(pred_img, label_img)
#     data["acc"] = Acc
#     data["conf"] = Conf
#     data["nums"] = Nums
#
#     with open("E:\\2020_SeoulChangeDetection\\Paper\\relability_diagram\\pred\\ensemble\\DRU_DA\\ensemble(m1~m10)\\dru_dist{0}_bin15.pickle".format(count), "wb") as f:
#         pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
#         print("Complete to write.")

# data["acc"] /= 2
# data["conf"] /= 2
# data["nums"] /= 2
# print(data["acc"])

# for pred_folder_nm in pred_list:
#     pred_folder_dir = os.path.join(pred_folders_dir, pred_folder_nm)
#     pred_file_list = os.listdir(pred_folder_dir)
#     data = {}
#
#     for pred_file_nm, label_file_nm in zip(pred_file_list, label_list) :
#         pred_file_dir = os.path.join(pred_folder_dir, pred_file_nm)
#         label_file_dir = os.path.join(label_folders_dir, label_file_nm)
#
#         pred_img = tifi.imread(pred_file_dir)
#         lbl_img = cv2.imread(label_file_dir, cv2.IMREAD_GRAYSCALE)
#
#         mask_1 = lbl_img >= 100
#         mask_0 = lbl_img < 100
#         lbl_img[mask_1] = 1
#         lbl_img[mask_0] = 0
#
#         Acc, Conf, Nums = build_diagram(pred_img, lbl_img)
#
#         data["acc"] = Acc
#         data["conf"] = Conf
#         data["nums"] = Nums
#
#         dist = os.path.splitext(label_file_nm)[0].split("_")[1]
#         type = os.path.splitext(pred_file_nm)[0].split("_")[0]
#         print(pred_folder_nm, pred_folder_nm+"_"+label_file_nm[:-4])
#         print(data["acc"])
#         with open("E:\\2020_SeoulChangeDetection\\Paper\\relability_diagram\\calib_pred\\{0}\\{1}.pickle".format(pred_folder_nm, pred_folder_nm+"_"+label_file_nm[:-4]), "wb") as f :
#             pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
#             # print("Complete to write {0}.pickle.".format(pred_folder_nm+"_"+label_file_nm[:-4]))

# # ECE calc
# pickle_dir = "E:\\2020_SeoulChangeDetection\\Paper\\histo_binning\\relability_diagram\\train"
# pickle_sub_folder_list = os.listdir(pickle_dir)
# for pickle_sub_folder_nm in pickle_sub_folder_list :
#     pickle_sub_dir = os.path.join(pickle_dir, pickle_sub_folder_nm)
#     pickle_file_list = os.listdir(pickle_sub_dir)
#     for pickle_file in pickle_file_list :
#         pickle_file_dir = os.path.join(pickle_sub_dir, pickle_file)
#         f = open(pickle_file_dir, 'rb')
#         p = pickle.load(f)
#         sum = np.sum(p["nums"])
#         ECE = 0
#         for acc, conf, num in zip(p["Acc"], p["Conf"], p["nums"]):
#             ECE = np.abs(acc-conf)*num/sum
#         print(pickle_file, ECE)

rd_pickle_dir = "E:\\2020_SeoulChangeDetection\\Paper\\relability_diagram\\pred\\ensemble\\DRU_DA"
dist_list = os.listdir(rd_pickle_dir)

x = np.arange(0, 1.0, 0.1)

for main_idx, dist in enumerate(dist_list) :
    dist_dir = os.path.join(rd_pickle_dir, dist)
    dist_list = os.listdir(dist_dir)
    for idx, pickle_nm in enumerate(dist_list) :
        pickle_dir = os.path.join(dist_dir, pickle_nm)
        f = open(pickle_dir, "rb")
        data = pickle.load(f)

        # ECE
        nums = data["nums"]
        sum = np.sum(nums)
        ECE = 0
        for acc, conf, num in zip(data["acc"], data["conf"], data["nums"]):
            sub_sum = np.abs(acc-conf)*num/sum
            ECE += sub_sum

        plot_idx = "95" + str(5*main_idx+idx+1)
        plt.subplot2grid((3,5), (main_idx, idx))
        plt.bar(data["conf"], data["acc"], color="b", alpha=0.5, width=0.05)
        plt.plot(data["conf"], data["acc"], color="b")
        plt.plot(x, x, color="r", linestyle="--")
        plt.title(pickle_nm.split(".")[0], fontdict={"fontsize" : 10})
        plt.text(0.5, 0.5, "ECE : {0}".format(str(ECE)))

plt.show()

# uncalib_pickle = "E:\\2020_SeoulChangeDetection\\Paper\\relability_diagram\\pred\\CA_DRU"
# uncalib_pickle = "E:\\2020_SeoulChangeDetection\\Paper\\relability_diagram\\calib_pred\\CA_DRU"
# calib_pickle = "E:\\2020_SeoulChangeDetection\\Paper\\relability_diagram\\calib_pred\\CA_DRU25"
#
# uncalib_list = os.listdir(uncalib_pickle)
# calib_list = os.listdir(calib_pickle)
#
# x = np.arange(0, 1.0, 0.1)
# idx = 0
# for uncalib_nm, calib_nm in zip(uncalib_list, calib_list) :
#
#     uncalib_dir = os.path.join(uncalib_pickle, uncalib_nm)
#     calib_dir = os.path.join(calib_pickle, calib_nm)
#
#     uncalib = open(uncalib_dir, "rb")
#     uncalib_data = pickle.load(uncalib)
#     calib = open(calib_dir, "rb")
#     calib_data = pickle.load(calib)
#
#     # ECE
#     un_nums = uncalib_data["nums"]
#     ca_nums = calib_data["nums"]
#     un_sum = np.sum(un_nums)
#     ca_sum = np.sum(ca_nums)
#
#     un_ECE = 0
#     ca_ECE = 0
#     for acc, conf, num in zip(uncalib_data["acc"], uncalib_data["conf"], uncalib_data["nums"]):
#         sub_sum = np.abs(acc-conf)*num/un_sum
#         un_ECE += sub_sum
#     for acc, conf, num in zip(calib_data["acc"], calib_data["conf"], calib_data["nums"]):
#         sub_sum = np.abs(acc-conf)*num/ca_sum
#         ca_ECE += sub_sum
#
#     # plot_idx = "55" + str(5*main_idx+idx+1)
#     plt.subplot2grid((5,2), (idx, 0))
#     plt.bar(uncalib_data["conf"], uncalib_data["acc"], color="b", alpha=0.5, width=0.05)
#     plt.plot(uncalib_data["conf"], uncalib_data["acc"], color="b")
#     plt.plot(x, x, color="r", linestyle="--")
#     plt.title(uncalib_nm.split(".")[0], fontdict={"fontsize" : 10})
#     plt.text(0.5, 0.5, "ECE : {0}".format(str(un_ECE)))
#
#     plt.subplot2grid((5, 2), (idx, 1))
#     plt.bar(calib_data["conf"], calib_data["acc"], color="b", alpha=0.5, width=0.05)
#     plt.plot(calib_data["conf"], calib_data["acc"], color="b")
#     plt.plot(x, x, color="r", linestyle="--")
#     plt.title(calib_nm.split(".")[0], fontdict={"fontsize": 10})
#     plt.text(0.5, 0.5, "ECE : {0}".format(str(ca_ECE)))
#
#     idx += 1
#
# plt.show()

# def histogram_binning(img_dir, pickle_results, bins=15):
#
#     pred_img = tifi.imread(img_dir)
#     pred_img_clone = np.zeros_like(pred_img)
#     acc = pickle_results["acc"]
#
#     for idx in range(bins):
#         sub_mask = (pred_img >= 0.0667 * idx) & (pred_img < 0.0667 * (1 + idx))
#         pred_img_clone[sub_mask] = acc[idx]
#     print(acc, np.unique(pred_img))
#     return pred_img_clone
#
# diagram_dir = "E:\\2020_SeoulChangeDetection\\Paper\\relability_diagram\\valid\\bin15"
# diagram_list = os.listdir(diagram_dir)
#
# pred_dir = "E:\\2020_SeoulChangeDetection\\Paper\\Pred_prob\\predict\\pred\\temp"
# pred_list = os.listdir(pred_dir)
#
# for diagram_nm, pred_model_nm in zip(diagram_list, pred_list) :
#
#     diagram_nm_dir = os.path.join(diagram_dir, diagram_nm)
#     pred_model_nm_dir = os.path.join(pred_dir, pred_model_nm)
#
#     pickle_nm = os.listdir(diagram_nm_dir)[0]
#     pickle_dir = os.path.join(diagram_nm_dir, pickle_nm)
#     f = open(pickle_dir, 'rb')
#     p = pickle.load(f)
#
#     pred_results_list = os.listdir(pred_model_nm_dir)
#     print(pickle_dir, pred_results_list)
#
#     for pred_nm in pred_results_list :
#         sub_pred_dir = os.path.join(pred_model_nm_dir, pred_nm)
#         calib_img = histogram_binning(sub_pred_dir, p)
#         tifi.imwrite("E:\\2020_SeoulChangeDetection\\Paper\\histo_binning\\modified_pred(valid_data)\\{0}\\{1}".format(diagram_nm, pred_nm), calib_img)
#         print(pred_nm)

# ca_pred_all_dir = "E:\\2020_SeoulChangeDetection\\Paper\\Pred_results(Naive)\\DA_pred\\DA_pred_all"
# calib_pred_dir = "E:\\2020_SeoulChangeDetection\\Paper\\histo_binning\\modified_pred(valid_data)\\DA_DRU"
# #uncalib_pred_dir = "E:\\2020_SeoulChangeDetection\\Paper\\Pred_prob\\predict\\pred\\DA_DRU\\DA_DRU"
#
# ca_pred_list = os.listdir(ca_pred_all_dir)
# calib_pred_list = os.listdir(calib_pred_dir)
# #uncalib_pred_list = os.listdir(uncalib_pred_dir)
#
# for ca_pred_nm, calib_pred_nm in zip(ca_pred_list, calib_pred_list):
#
#     ca_pred_dir = os.path.join(ca_pred_all_dir, ca_pred_nm)
#     calib_dir = os.path.join(calib_pred_dir, calib_pred_nm)
#
#     pred_img = cv2.imread(ca_pred_dir, cv2.IMREAD_UNCHANGED)
#     b = pred_img[:, :, 0]
#     g = pred_img[:, :, 1]
#     r = pred_img[:, :, 2]
#
#     mask255 = b > 125
#     mask0 = b <= 125
#     b[mask255] = 255
#     b[mask0] = 0
#     mask255 = g > 125
#     mask0 = g <= 125
#     g[mask255] = 255
#     g[mask0] = 0
#     mask255 = r > 125
#     mask0 = r <= 125
#     r[mask255] = 255
#     r[mask0] = 0
#
#     pred_img = cv2.merge((b, g, r))
#     pred_sum = np.sum(pred_img, axis=-1)
#     error_mask = pred_sum == 255
#     error_count = len(pred_sum[error_mask])
#     print("Total Pixel : ", pred_img.shape[0]*pred_img.shape[1])
#     print("Error count : ", error_count)
#
#     calib_img = tifi.imread(calib_dir)
#     check_idx = (calib_img >= 0.45) & (calib_img < 0.55)
#     check_target = pred_img[check_idx]
#     check_target_sum = np.sum(check_target, axis=-1)
#     print("Check target : ", len(check_target))
#     check_error_mask = check_target_sum == 255
#     check_error_count = len(check_target_sum[check_error_mask])
#     print("correct check target : ", check_error_count/len(check_target))