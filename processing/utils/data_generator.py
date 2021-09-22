"""
Faster R-CNN from 21.07.29.
"""
import cv2
import math
import copy
import random
import numpy as np

def augmentation(img_data, config, function):

    img_dir = img_data["filepath"]
    img = cv2.imread(img_dir)
    height, width = img.shape[0], img.shape[1]

    if function :
        if config.horizontal_flips and np.random.randint(0,1)==1 :
            img = cv2.flip(img, 0)
            for bbox in img_data["bboxes"]:
                y1 = bbox["y1"]
                y2 = bbox["y2"]
                bbox["y1"] = height - y1
                bbox["y2"] = height - y2

        if config.vertical_flips and np.random.randint(0,1)==1 :
            img = cv2.flip(img, 1)
            for bbox in img_data["bboxes"]:
                x1 = bbox["x1"]
                x2 = bbox["x2"]
                bbox["x1"] = width - x1
                bbox["x2"] = width - x2

        if config.rot :
            angle = np.random.choice([0, 90, 180, 270], 1)[0]
            if angle == 270:
                img = np.transpose(img, (1, 0, 2))
                img = cv2.flip(img, 0)
            elif angle == 180:
                img = cv2.flip(img, -1)
            elif angle == 90:
                img = np.transpose(img, (1, 0, 2))
                img = cv2.flip(img, 1)
            elif angle == 0:
                pass
            for bbox in img_data['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                y1 = bbox['y1']
                y2 = bbox['y2']
                if angle == 270:
                    bbox['x1'] = y1
                    bbox['x2'] = y2
                    bbox['y1'] = width - x2
                    bbox['y2'] = width - x1
                elif angle == 180:
                    bbox['x2'] = width - x1
                    bbox['x1'] = width - x2
                    bbox['y2'] = height - y1
                    bbox['y1'] = height - y2
                elif angle == 90:
                    bbox['x1'] = height - y2
                    bbox['x2'] = height - y1
                    bbox['y1'] = x1
                    bbox['y2'] = x2
                elif angle == 0:
                    pass

    img_data["height"] = img.shape[0]
    img_data["width"] = img.shape[1]
    return img_data, img

def output_size(height, width, downscale):
    return math.ceil(height/downscale), math.ceil(width/downscale)

# def iou(in_range_anchors, gt_box):
#
#     iou_array = np.zeros([in_range_anchors.shape[0], gt_box.shape[0]])
#     for idx, in_range_anchor in enumerate(in_range_anchors):
#         iou_x1 = np.maximum(in_range_anchor[0], gt_box[:,0])
#         iou_y1 = np.maximum(in_range_anchor[1], gt_box[:,1])
#         iou_x2 = np.minimum(in_range_anchor[2], gt_box[:,2])
#         iou_y2 = np.minimum(in_range_anchor[3], gt_box[:,3])
#
#         h = np.maximum(0.0, iou_y2-iou_y1)
#         w = np.maximum(0.0, iou_x2-iou_x1)
#
#         inter = h*w
#         union = (in_range_anchor[2]-in_range_anchor[0])*(in_range_anchor[3]-in_range_anchor[1])+\
#                 (gt_box[:,2] - gt_box[:,0])*(gt_box[:,3]-gt_box[:,1]) - inter + np.finfo(np.float32).eps
#         iou = inter / union
#         iou_array[idx, :] = iou
#
#     return iou_array

def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w*h


def iou(a, b):
    # a and b should be (x1,y1,x2,y2)
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0
    area_i = intersection(a, b)
    area_u = union(a, b, area_i)
    return float(area_i) / float(area_u + 1e-6)

def format_loc(anchors, base_anchors):

    height = anchors[:, 2] - anchors[:, 0]
    width = anchors[:, 3] - anchors[:, 1]
    ctr_y = anchors[:, 0] + height*0.5
    ctr_x = anchors[:, 1] + width*0.5

    base_height = base_anchors[:, 2] - base_anchors[:, 0]
    base_width = base_anchors[:, 3] - base_anchors[:, 1]
    base_ctr_y = base_anchors[:, 0] + base_height*0.5
    base_ctr_x = base_anchors[:, 1] + base_width*0.5

    eps = np.finfo(np.float32).eps
    height = np.maximum(eps, height)
    width = np.maximum(eps, width)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    anchor_loc_target = np.stack((dy, dx, dh, dw), axis=1)
    return anchor_loc_target

def get_new_img_size(height, width, img_min_side=600):

    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)
    return resized_width, resized_height

def gt_calc_mix(img_data, config, height, width, resized_height, resized_width):

    anchor_ratio = np.array(config.anchor_ratio) # [0, 1, 2]
    anchor_scale = np.array(config.anchor_scale) # [0, 1, 2]
    downscale = config.downscale_ratio

    output_height, output_width = output_size(height, width, downscale)

    # 투영하지 않은 사이즈 기준 진행
    # Ground truth bbox extraction
    bbox_len = len(img_data["bboxes"])
    gt_bbox = np.zeros([bbox_len, 4]) # shape : [bbox_len, 4]
    for idx, bbox in enumerate(img_data["bboxes"]) :
        gt_bbox[idx, 0] = bbox["x1"] * (resized_width / float(width))
        gt_bbox[idx, 1] = bbox["y1"] * (resized_height / float(height))
        gt_bbox[idx, 2] = bbox["x2"] * (resized_width / float(width))
        gt_bbox[idx, 3] = bbox["y2"] * (resized_height / float(height))

    # Generating base anchor boxes
    num_anchor = len(anchor_ratio)*len(anchor_scale)
    base_anchor_bbox = np.zeros([num_anchor, 4]) # shape : [anchor_num, 4]
    output_height_index = downscale*(np.arange(output_height)+0.5)
    output_width_index = downscale*(np.arange(output_width)+0.5)

    index_array = np.zeros([len(output_height_index),
                            len(output_width_index),
                            2])
    for idx in range(0, len(output_height_index)):
        index_array[idx, :, 0] = output_height_index[idx]
        index_array[idx, :, 1] = output_width_index

    for idx, scale in enumerate(anchor_scale) :
        anchor_height = downscale*scale*np.sqrt(anchor_ratio)
        anchor_width = downscale*scale/np.sqrt(anchor_ratio)
        x1 = -anchor_width/2
        x2 = anchor_width/2
        y1 = -anchor_height/2
        y2 = anchor_height/2
        base_anchor_bbox[idx*len(anchor_ratio):(idx+1)*len(anchor_ratio), 0] = x1
        base_anchor_bbox[idx*len(anchor_ratio):(idx+1)*len(anchor_ratio), 1] = y1
        base_anchor_bbox[idx*len(anchor_ratio):(idx+1)*len(anchor_ratio), 2] = x2
        base_anchor_bbox[idx*len(anchor_ratio):(idx+1)*len(anchor_ratio), 3] = y2

    anchor_box_coord_array = np.zeros([index_array.shape[0], index_array.shape[1], num_anchor, 4])

    for h_idx in range(index_array.shape[0]):
        for w_idx in range(index_array.shape[1]):
            anchor_box_coord_array[h_idx, w_idx] = (index_array[h_idx, w_idx] + base_anchor_bbox.reshape(-1, 2, 2)).reshape(-1, 4)

    anchor_box_coord_array = anchor_box_coord_array.reshape(-1, 4)

    ## labeling anchor boxes based on gt_boxes from here

    # range in the image
    in_range_index = np.where((anchor_box_coord_array[:,0] >= 0)
                              &(anchor_box_coord_array[:,1] >= 0)
                              &(anchor_box_coord_array[:,2] <= width)
                              &(anchor_box_coord_array[:,3] <= height))[0]

    valid_anchors = anchor_box_coord_array[in_range_index]
    iou_array = iou(in_range_anchors=valid_anchors, gt_box=gt_bbox)
    # positive / negative / neither
    # over 70% / under 30%
    max_iou_of_gts = np.max(iou_array, axis=1)

    positive_label = np.where(max_iou_of_gts >= config.positive_thr)[0]
    negative_label = np.where(max_iou_of_gts < config.negative_thr)[0]
    valid_labels = np.empty((in_range_index.shape[0],), dtype=np.int32)
    valid_labels.fill(-1)
    valid_labels[positive_label]=1
    valid_labels[negative_label]=0

    # max iou for each gt
    max_iou_of_anchor = np.max(iou_array, axis=0)
    max_iou_of_anchor_label = np.where(iou_array == max_iou_of_anchor)[0]
    valid_labels[max_iou_of_anchor_label] = 1

    # setting positive / negative ratio
    total_n_pos = len(np.where(valid_labels == 1)[0])
    n_pos_sample = config.num_samples * config.pos_ratio if total_n_pos > config.num_samples * config.pos_ratio else total_n_pos
    n_neg_sample = config.num_samples - n_pos_sample

    pos_index = np.where(valid_labels == 1)[0]
    if len(pos_index) > config.num_samples * config.pos_ratio:
        disable_index = np.random.choice(pos_index, size=len(pos_index) - n_pos_sample, replace=False)
        valid_labels[disable_index] = -1

    neg_index = np.where(valid_labels == 0)[0]
    disable_index = np.random.choice(neg_index, size=len(neg_index) - n_neg_sample, replace=False)
    valid_labels[disable_index] = -1

    argmax_iou = np.argmax(iou_array, axis=1)
    max_iou_box = gt_bbox[argmax_iou]
    anchor_loc_format_target = format_loc(valid_anchors, max_iou_box)

    anchor_target_labels = np.empty((len(anchor_box_coord_array),), dtype=np.int32)
    anchor_target_format_locations = np.zeros((len(anchor_box_coord_array), 4), dtype=np.float32)

    anchor_target_labels.fill(-1)
    anchor_target_labels[in_range_index] = valid_labels

    anchor_target_format_locations[in_range_index] = anchor_loc_format_target
    return anchor_target_labels, anchor_target_format_locations

def gt_calc_origin(img_data, config, height, width, resized_height, resized_width):

    downscale = float(config.downscale_ratio)
    anchor_sizes = config.anchor_box_scales
    anchor_ratios = config.anchor_box_ratios
    num_anchors = len(anchor_sizes)*len(anchor_ratios)

    output_width, output_height = output_size(height, width, downscale)
    n_anchratios = len(anchor_ratios)

    # initialise empty output objectives
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

    num_bboxes = len(img_data["bboxes"])

    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1*np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    gta = np.zeros((num_bboxes, 4))
    for bbox_num, bbox in enumerate(img_data["bboxes"]):
        gta[bbox_num, 0] = bbox["x1"]*(resized_width / float(width))
        gta[bbox_num, 1] = bbox["x2"]*(resized_width / float(width))
        gta[bbox_num, 2] = bbox["y1"]*(resized_height / float(height))
        gta[bbox_num, 3] = bbox["y2"]*(resized_height / float(height))

    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(n_anchratios):
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]

            for ix in range(output_width):
                x1_anc = downscale * (ix + 0.5) - anchor_x/2
                x2_anc = downscale * (ix + 0.5) + anchor_x/2

                if x1_anc < 0 or x2_anc > resized_width :
                    continue

                for jy in range(output_height):
                    y1_anc = downscale * (jy + 0.5) - anchor_y/2
                    y2_anc = downscale * (jy + 0.5) + anchor_y/2
                    if y1_anc < 0 or y2_anc > resized_height :
                        continue

                    bbox_type = "neg"
                    best_iou_for_loc = 0.0

                    for bbox_num in range(num_bboxes):
                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]],
                                       [x1_anc, y1_anc, x2_anc, y2_anc])

                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > config.positive_thr:
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                            cxa = (x1_anc + x2_anc) / 2.0
                            cya = (y1_anc + y2_anc) / 2.0

                            tx = (cx - cxa) / (x2_anc - x1_anc)
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))

                        if img_data["bboxes"][bbox_num]["class"] != "background" :
                            if curr_iou > best_iou_for_bbox[bbox_num] :
                                best_iou_for_bbox[bbox_num] = curr_iou
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                best_x_for_bbox[bbox_num, :] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                best_dx_for_bbox[bbox_num, :] = [tx, ty, tw, th]

                            if curr_iou > config.positive_thr:
                                bbox_type = "pos"
                                num_anchors_for_bbox[bbox_num] += 1
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)

                            if config.negative_thr < curr_iou < config.positive_thr :
                                if bbox_type != "pos":
                                    bbox_type = "neutral"

                        if bbox_type == "neg":
                            y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                            y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        elif bbox_type == 'neutral':
                            y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                            y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        elif bbox_type == 'pos':
                            y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                            y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                            start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
                            y_rpn_regr[jy, ix, start:start + 4] = best_regr

    for idx in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[idx] == 0 : # 어떤 GT box에 대하여 어느 한 개의 anchor 박스도 positive가 되지 않은 경우
            if best_anchor_for_bbox[idx, 0] == -1: # iou가 0보다 큰 경우가 없으면 제외
                continue
            y_is_box_valid[
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios*best_anchor_for_bbox[idx,3]
            ] = 1
            y_rpn_overlap[
                best_anchor_for_bbox[idx,0], best_anchor_for_bbox[idx,1], best_anchor_for_bbox[idx,2] + n_anchratios*best_anchor_for_bbox[idx,3]
            ] = 1
            start = 4 * (best_anchor_for_bbox[idx, 2] + n_anchratios * best_anchor_for_bbox[idx, 3])
            y_rpn_regr[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], start:start + 4
            ] = best_dx_for_bbox[idx, :]

    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_pos = len(pos_locs[0])

    # one issue is that the RPN has many more negative than positive regions, so we turn off some of the negative
    # regions. We also limit it to 256 regions.
    num_regions = 256

    if len(pos_locs[0]) > num_regions / 2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions / 2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions / 2

    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)

def get_gt_featuremap(all_img_data, config, augment=True, mode="train"):

    while True :
        if mode == "train" : random.shuffle(all_img_data)
        for img_data in all_img_data :
            if augment :
                aug_img_data, x_img = augmentation(img_data=img_data,
                                                   config=config,
                                                   function=True)
            else :
                aug_img_data, x_img = augmentation(img_data=img_data,
                                                   config=config,
                                                   function=False)

            height, width = aug_img_data["height"], aug_img_data["width"]
            (resized_width, resized_height) = get_new_img_size(height, width)
            x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
            try :
                width, height = x_img.shape[0], x_img.shape[1]
                gt_cls, gt_regr = gt_calc_origin(aug_img_data, config, height, width, resized_height, resized_width)
            except :
                pass

            x_img = x_img[:, :, (2, 1, 0)]
            x_img = np.transpose(x_img, (2, 0, 1))
            x_img = np.expand_dims(x_img, axis=0)

            x_img = np.transpose(x_img, (0, 2, 3, 1))
            gt_cls = np.transpose(gt_cls, (0, 2, 3, 1))
            gt_regr = np.transpose(gt_regr, (0, 2, 3, 1))

            yield np.copy(x_img), [np.copy(gt_cls), np.copy(gt_regr)], aug_img_data

if __name__ == "__main__" :
    pass