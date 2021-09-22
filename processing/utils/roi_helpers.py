import numpy as np
import math
import copy

def calc_iou(R, img_data, config, class_mapping):
    """
    :param R: all_boxes
    :param img_data:
    :param config:
    :param class_mapping:
    :return:
    """
    bboxes = img_data["bboxes"]
    (width, height) = (img_data["width"], img_data["height"])

    # get image dimensions for resizing
    (resized_width, resized_height) = data_generators.get_new_img_size(width, height, C.im_size)

    gta = np.zeros((len(bboxes), 4))

    for bbox_num, bbox in enumerate(bboxes):
        # get the GT box coordinates, and resize to account for image resizing
        gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width)) / config.downsalce_ratio))
        gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width)) / config.downsalce_ratio))
        gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height)) / config.downsalce_ratio))
        gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height)) / config.downsalce_ratio))

    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = []

    for ix in range(R.shape[0]):
        (x1, y1, x2, y2) = R[ix, :]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        best_iou = 0.0
        best_bbox = -1

        for bbox_num in range(len(bboxes)):
            curr_iou = data_generators.iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]],
                                           [x1, y1, x2, y2])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num # 가장 큰 iou를 갖는 gbox를 선정

            if best_iou < config.negative_thr:
                continue
            else:
                w = x2-x1
                h = y2-y1
                x_roi.append([x1, y1, w, h])
                IoUs.append(best_iou)

                if config.negative_thr <= best_iou < config.positive_thr:
                    cls_name = "background"
                elif config.positive_thr <= best_iou :
                    cls_name = bboxes[best_bbox]["class"]
                    cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
                    cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

                    cx = x1 + w / 2.0
                    cy = y1 + h / 2.0

                    tx = (cxg - cx) / float(w)
                    ty = (cyg - cy) / float(h)
                    tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
                    th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
                else :
                    print('roi = {}'.format(best_iou))
                    raise RuntimeError

            class_num = class_mapping[cls_name]
            class_label = len(class_mapping) * [0]
            class_label[class_num] = 1 # one_hot_encoding
            y_class_num.append(copy.deepcopy(class_label))
            coords = [0] * 4 * (len(class_mapping) - 1)
            labels = [0] * 4 * (len(class_mapping) - 1)

            if cls_name != "background":
                label_pos = 4 * class_num
                sx, sy, sw, sh = config.classifier_regr_std
                coords[label_pos:4 + label_pos] = [sx * tx, sy * ty, sw * tw, sh * th]
                labels[label_pos:4 + label_pos] = [1, 1, 1, 1]
                y_class_regr_coords.append(copy.deepcopy(coords))
                y_class_regr_label.append(copy.deepcopy(labels))
            else:
                y_class_regr_coords.append(copy.deepcopy(coords))
                y_class_regr_label.append(copy.deepcopy(labels))

            if len(x_roi) == 0:
                return None, None, None, None

            X = np.array(x_roi)
            Y1 = np.array(y_class_num)
            Y2 = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)], axis=1)

            return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs




def apply_regr_np(X, T):
    """
    :param X: A[:, :, :, curr_layer]
    :param T: regr
    :return:
    """
    try :
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w/2
        cy = y + h/2
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = np.exp(tw.astype(np.float64))*w
        h1 = np.exp(th.astype(np.float64))*h
        x1 = cx1 - w1/2.
        y1 = cy1 - h1/2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])

    except Exception as e:
        print(e)
        return X

def non_max_suppression_fast(boxes, probs, overlap_threshold=0.9, max_boxes=300):
    """
    :param boxes: all_boxes
    :param probs: all_probs
    :param overlap_threshold:
    :param max_boxes:
    :return:
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes from min to max
    idxs = np.argsort(probs)

    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int / (area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_threshold)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    print(boxes)
    print(probs)
    return boxes, probs

def rpn_to_roi(rpn_layer, regr_layer, config,
               use_regr=True, max_boxes=300, overlap_threshold=0.9):
    # p_rpn[0](rpn_layer) : pred cls feature
    # p_rpn[1](regr_layer) : pred regr feature - h x w x c (4*num_of_anchors)
    # regr_layer = regr_layer / config.std_scaling

    anchor_sizes = config.anchor_box_scales
    anchor_ratios = config.anchor_box_ratios

    (height, width) = rpn_layer.shape[1:3]

    curr_layer = 0
    A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3])) # rpn_layer.shape[3] = num_of_anchor

    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:

            anchor_x = (anchor_size * anchor_ratio[0]) / config.downscale_ratio
            anchor_y = (anchor_size * anchor_ratio[1]) / config.downscale_ratio

            regr = regr_layer[0, :, :, 4*curr_layer:4*curr_layer+4]
            regr = np.transpose(regr, (2, 0, 1))

            X, Y = np.meshgrid(np.arange(width), np.arange(height))

            A[0, :, :, curr_layer] = X - anchor_x/2 # 좌상단 x
            A[1, :, :, curr_layer] = Y - anchor_y/2 # 좌상단 y
            A[2, :, :, curr_layer] = anchor_x # anchor 박스의 width
            A[3, :, :, curr_layer] = anchor_y # anchor 박스의 height

            if use_regr :
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

            # 1과 A[2,:,:,curr_layer] 중 큰 요소를 선택, 너비가 1보다 큰 값, 즉 최소크기 = downscale
            A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
            A[2, :, :, curr_layer] += A[0, :, :, curr_layer] # 우하단 x
            A[3, :, :, curr_layer] += A[1, :, :, curr_layer] # 우하단 y

            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer]) # 좌상단 x 좌표가 0보다 작지 않도록 clip
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer]) # 좌상단 y 좌표가 0보다 작지 않도록 clip
            A[2, :, :, curr_layer] = np.minimum(width, A[2, :, :, curr_layer]) # 우하단 x 좌표가 범위를 넘어가지 않도록 clip
            A[3, :, :, curr_layer] = np.minimum(height, A[2, :, :, curr_layer]) # 우하단 y 좌표가 범위를 넘어가지 않도록 clip

            curr_layer += 1

        all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))
        all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))

        x1 = all_boxes[:, 0]
        y1 = all_boxes[:, 1]
        x2 = all_boxes[:, 2]
        y2 = all_boxes[:, 3]

        idxs = np.where((x1-x2 >= 0) | (y1-y2 >= 0))
        all_boxes = np.delete(all_boxes, idxs, 0)
        all_probs = np.delete(all_probs, idxs, 0)

        result = non_max_suppression_fast(all_boxes, all_probs, overlap_threshold=overlap_threshold, max_boxes=max_boxes)[0]
        return result