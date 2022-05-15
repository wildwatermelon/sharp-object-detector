import sys
import os
import glob
import xml.etree.ElementTree as ET
import argparse
import json
from src.dataset_const import *
def get_args():
    parser = argparse.ArgumentParser("You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--year", type=str, default="2012", help="The year of dataset (2007 or 2012)")
    parser.add_argument("--data_path", type=str, default="../sharp_object_dataset/", help="the root folder of dataset")
    parser.add_argument("--test_set", type=str, default="val")

    args = parser.parse_args()

    return args

def generate(opt):
    input_list_path = os.path.join(opt.data_path, "ImageSets/Main/{}.txt".format(opt.test_set))
    input_list_anno_path = os.path.join(opt.data_path, "Annotations".format(opt.test_set))
    print(input_list_path)
    print(input_list_anno_path)
    image_ids = (
        open(input_list_path).read().strip().split()
    )

    if not os.path.exists("./input"):
        os.makedirs("./input")
    if not os.path.exists("./input/ground-truth"):
        os.makedirs("./input/ground-truth")

    for image_id in image_ids:
        with open("./input/ground-truth/" + image_id + "_ground_truth.json", "w") as outfile:
            bounding_boxes = []
            root = ET.parse(
                input_list_anno_path + '/' + image_id + ".xml"
            ).getroot()
            for obj in root.findall("object"):
                if obj.find("difficult") != None:
                    difficult = obj.find("difficult").text
                    if int(difficult) == 1:
                        continue
                obj_name = obj.find("name").text
                bndbox = obj.find("bndbox")
                left = bndbox.find("xmin").text
                top = bndbox.find("ymin").text
                right = bndbox.find("xmax").text
                bottom = bndbox.find("ymax").text
                bbox = left + " " + top + " " + right + " " + bottom
                bounding_boxes.append(
                    {"class_name": obj_name, "bbox": bbox, "used": False}
                )
            json.dump(bounding_boxes, outfile)
    print("Conversion completed!")

if __name__ == "__main__":
    opt = get_args()
    generate(opt)