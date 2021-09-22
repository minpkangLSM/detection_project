"""
2021.07.28. - Pascal parser for Faster-RCNN directly implemented.
Kangmin Park, Lab. for Sensor and Modeling, Univ. of Seoul.
"""
import os
import xml.etree.ElementTree as ET

def parser(pascal_dir):

    all_imgs = []
    classes_count = {}
    classes_mapping = {}

    pascal_dir = os.path.join(pascal_dir, "VOC2012")
    annot_dir = os.path.join(pascal_dir, "Annotations")
    img_dir = os.path.join(pascal_dir, "JPEGImages")

    imgsets_path_trainval = os.path.join(pascal_dir, "ImageSets", "Main", "train_val.txt")
    imgsets_path_train = os.path.join(pascal_dir, "ImageSets", "Main", "train.txt")
    imgsets_path_val = os.path.join(pascal_dir, "ImageSets", "Main", "val.txt")

    trainval_files = []
    train_files = []
    val_files = []

    with open(imgsets_path_trainval) as f:
        for line in f:
            trainval_files.append(line.strip() + ".jpg")
    with open(imgsets_path_train) as f:
        for line in f:
            train_files.append(line.strip() + ".jpg")
    with open(imgsets_path_val) as f:
        for line in f:
            val_files.append(line.strip() + ".jpg")

    annots = [os.path.join(annot_dir, annot_file) for annot_file in os.listdir(annot_dir)]
    idx = 0
    for annot in annots:
        idx += 1
        exist_flag = False

        et = ET.parse(annot)
        element = et.getroot()

        element_filenm = element.find("filename").text
        element_objs = element.findall("object")
        element_height = int(element.find("size").find("height").text)
        element_width = int(element.find("size").find("width").text)

        if len(element_objs) > 0:
            annotation_data = {"filepath": os.path.join(img_dir, element_filenm),
                               "height": element_height,
                               "width": element_width,
                               "bboxes": [],
                               "image_id": idx}

            if element_filenm in trainval_files:
                annotation_data["imageset"] = "trainval"
                exist_flag = True
            if element_filenm in train_files:
                annotation_data["imageset"] = "train"
                exist_flag = True
            if element_filenm in val_files:
                annotation_data["imageset"] = "val"
                exist_flag = True
        if not exist_flag: continue

        for element_obj in element_objs:
            class_name = element_obj.find("name").text
            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in classes_mapping:
                classes_mapping[class_name] = len(classes_mapping)

            obj_bbox = element_obj.find("bndbox")
            x1 = int(round(float(obj_bbox.find("xmin").text)))
            x2 = int(round(float(obj_bbox.find("xmax").text)))
            y1 = int(round(float(obj_bbox.find("ymin").text)))
            y2 = int(round(float(obj_bbox.find("ymax").text)))
            difficulty = int(element_obj.find("difficult").text) == 1
            annotation_data["bboxes"].append(
                {"class": class_name,
                 "x1": x1,
                 "y1": y1,
                 "x2": x2,
                 "y2": y2,
                 "difficult": difficulty}
            )
        all_imgs.append(annotation_data)

    return all_imgs, classes_count, classes_mapping

if __name__ == "__main__" :

    data_dir = "E:\\Deep_learning_dataset\Pascal_2012\VOCtrainval_11-May-2012\VOCdevkit"
    all_imgs, classes_count, classes_mapping = parsing_pascal(pascal_dir = data_dir)