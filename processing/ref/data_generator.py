import cv2
import numpy as np

def augmentation(image, img_info, set):
    """

    :param image: image instance from cv2.imread
    :param bboxes: dict type in list
    :param set: config setting
    :return:
    """
    height, width = image.shape[:2]

    if set.augmentation_bool :
        # cv2.flip : 0 - horizontal flip, 1 - vertical flip
        if set.horizontal_bool and np.random.randint(2):
            x_img = cv2.flip(image,0)
            for idx in range(len(img_info["bboxes"])):
                img_info["bboxes"][idx]["y1"] = height - img_info["bboxes"][idx]["y2"]
                img_info["bboxes"][idx]["y2"] = height - img_info["bboxes"][idx]["y1"]

        if set.vertical_bool and np.random.randint(2):
            x_img = cv2.flip(image,1)
            for idx in range(len(img_info["bboxes"])):
                img_info["bboxes"][idx]["x1"] = width - img_info["bboxes"][idx]["x2"]
                img_info["bboxes"][idx]["x2"] = width - img_info["bboxes"][idx]["x1"]

        # cv2.rotate : 0
        if set.rotate_bool and np.random.randint(2):
            rot = np.random.choice([0, 90, 180, 270], 1)
            if rot == 0:
                pass
            elif rot == 90:
                x_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                img_info["height"] = img_info["width"]
                img_info["width"] = img_info["height"]
                for idx in range(len(img_info["bboxes"])):
                    img_info["bboxes"][idx]["x1"] = height - img_info["bboxes"][idx]["y2"]
                    img_info["bboxes"][idx]["y1"] = width - img_info["bboxes"][idx]["x1"]
                    img_info["bboxes"][idx]["x2"] = height - img_info["bboxes"][idx]["y1"]
                    img_info["bboxes"][idx]["y2"] = width - img_info["bboxes"][idx]["x2"]
            elif rot == 180:
                x_img = cv2.rotate(image, cv2.ROTATE_180)
                for idx in range(len(img_info["bboxes"])):
                    img_info["bboxes"][idx]["x1"] = width - img_info["bboxes"][idx]["x2"]
                    img_info["bboxes"][idx]["y1"] = height - img_info["bboxes"][idx]["y2"]
                    img_info["bboxes"][idx]["x2"] = width - img_info["bboxes"][idx]["y1"]
                    img_info["bboxes"][idx]["y2"] = height - img_info["bboxes"][idx]["y1"]
            elif rot == 270:
                x_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_info["height"] = img_info["width"]
                img_info["width"] = img_info["height"]
                for idx in range(len(img_info["bboxes"])):
                    img_info["bboxes"][idx]["x1"] = img_info["bboxes"][idx]["y1"]
                    img_info["bboxes"][idx]["y1"] = width - img_info["bboxes"][idx]["x2"]
                    img_info["bboxes"][idx]["x2"] = img_info["bboxes"][idx]["y2"]
                    img_info["bboxes"][idx]["y2"] = width - img_info["bboxes"][idx]["x1"]
    else :
        x_img = image
        img_info = img_info

    return x_img, img_info

def base_anchor(ratio, scale, downscale):

    anchor_num = len(ratio)*len(scale)
    anchor_frame = np.zeros((anchor_num, 3))
    for idx, scale in enumerate(scale):
        height = downscale * scale * np.sqrt(ratio)
        width = downscale * scale / np.sqrt(ratio)
        x1 = -width/2
        y1 = -height/2
        x2 = width/2
        y2 = height/2
        anchor_frame[idx * len(ratio):(idx + 1) * len(ratio), 0] = x1
        anchor_frame[idx * len(ratio):(idx + 1) * len(ratio), 1] = y1
        anchor_frame[idx * len(ratio):(idx + 1) * len(ratio), 2] = x2
        anchor_frame[idx * len(ratio):(idx + 1) * len(ratio), 3] = y2
    return anchor_frame

def iou(anchor_boxes, bboxes):
    """

    :param anchor_boxes: [x1, y1, x2, y2], shape : [-1, 4]
    :param bboxes: [x1, y1, x2, y2], shape : [-1, 4]
    :return:
    """
    ious = np.zeros((anchor_boxes.shape[0], bboxes.shape[0]))
    for idx, anchor_box in enumerate(anchor_boxes):
        x1_max = np.maximum(anchor_box[0], bboxes[:,0])
        y1_max = np.maximum(anchor_box[1], bboxes[:,1])
        x2_min = np.minimum(anchor_box[2], bboxes[:,2])
        y2_min = np.minimum(anchor_box[3], bboxes[:,3])

        inter_h = np.maximum(0.0, y2_min - y1_max)
        inter_w = np.maximum(0.0, x2_min - x1_max)

        inter = inter_h*inter_w
        union = (anchor_box[2]-anchor_box[0])*(anchor_box[3]-anchor_box[1]) + (bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])-inter+0.00001
        iou = inter/union
        ious[idx, :] = iou
    return ious


def gt_calc(img_info, set):

    anchor_ratio = set.anchor_box_ratio
    anchor_scale = set.anchor_box_scale
    anchor_num = len(anchor_ratio)*len(anchor_scale)
    downscale = set.downscale_ratio
    height, width = img_info["height"], img_info["width"]
    f_height, f_width = int(height/downscale), int(width/downscale)

    # Setting bounding boxes
    bboxes = img_info["bboxes"]
    bbox_coords = np.zeros((len(bboxes), 4))  # do not need to consider the class of bounding box in rpn part
    for idx in range(len(bboxes)):
        bbox_coords[idx, :] = bboxes[idx][0], bboxes[idx][1], bboxes[idx][2], bboxes[idx][3]

    # Setting anchor boxes
    base_anchors = base_anchor(ratio=anchor_ratio,
                                scale=anchor_scale,
                                downscale=downscale)
    ctr_x = np.arange(downscale / 2 + 0.5, width, downscale)
    ctr_y = np.arange(downscale / 2 + 0.5, height, downscale)
    ctr_coords = np.zeros((f_height, f_width, 2))
    for idx, y in enumerate(ctr_y):
        ctr_coords[idx, :, 0] = ctr_x
        ctr_coords[idx, :, 1] = y

    anchor_coords = np.zeros((f_height, f_width, anchor_num, 4)) # size based on original image size (not output feature)
    for x_idx in range(f_width):
        for y_idx in range(f_width):
            x1 = ctr_coords[x_idx, y_idx, 0] + base_anchors[:, 0]
            y1 = ctr_coords[x_idx, y_idx, 1] + base_anchors[:, 1]
            x2 = ctr_coords[x_idx, y_idx, 0] + base_anchors[:, 2]
            y2 = ctr_coords[x_idx, y_idx, 1] + base_anchors[:, 3]
            anchor_coords[x_idx, y_idx] = np.stack([x1, y1, x2, y2], axis=-1)
    anchor_coords = anchor_coords.reshape(-1,4) # reshape anchor_coords array, which is identical shape with bbox_coords shape

    # clip
    in_range = np.where((anchor_coords[:,0] >= 0)
                        &(anchor_coords[:,1] >= 0)
                        &(anchor_coords[:,2] >= width)
                        &(anchor_coords[:,3] >= height))
    print(bbox_coords.shape)
    print(anchor_coords.shape)
    # valid_anchor_coords = anchor_coords[in_range]
    # valid_labels = np.empty((valid_anchor_coords.shape[0],), dtype=np.int32)
    #
    # # iou : positive or negative
    # ious = iou(valid_anchor_coords, bbox_coords)
    # max_ious = np.max(ious, axis=-1)
    # max_ious_idx = np.argmax(ious, axis=-1)
    #
    # # Option A : over 70%
    # positive_idx_A = np.where(max_ious >= set.positive_thr)
    # negative_idx_A = np.where(max_ious < set.negative_thr)
    #
    # # Option B : best iou for each bbox
    # best_iou_for_bbox = np.max(ious, axis=0)
    # positive_idx_B = best_iou_for_bbox == ious





    pass
    # return gt_regr, gt_cls

def gt_generator(all_imgs_info, set):
    """
    :param all_imgs: dictionary type.
    keys : 1) filepath(image directory), 2) height, 3) width, 4) bboxes, 5) image_id(ordering index), 6) imageset
    sub_key of bboxes : 1) class, 2) x1, 3) y1, 4) x2, 5) y2, 6) difficult(not necessary)
    :param set : config setting
    :return:
    """

    for img_info in all_imgs_info :

        # load X(img) data
        image = cv2.imread(img_info["filepath"])
        aug_x_img, aug_img_info = augmentation(image=image,
                                             img_info=img_info,
                                             set=set)
        gt_calc(img_info=aug_img_info,
                set=set)


    return None