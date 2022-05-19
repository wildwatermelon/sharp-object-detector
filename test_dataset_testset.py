import os
import argparse
import shutil
import cv2
import numpy as np
from src.utils import *
import pickle
from src.yolo_net import Yolo as Yolo
from src.dataset_const import *
import json
from src.dataset_const import *

def get_args():
    parser = argparse.ArgumentParser("You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--image_size", type=int, default=448, help="The common width and height for all images")
    # parser.add_argument("--conf_threshold", type=float, default=0.35)
    # parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--conf_threshold", type=float, default=0.35)
    parser.add_argument("--nms_threshold", type=float, default=0.35)
    parser.add_argument("--test_set", type=str, default="test")
    parser.add_argument("--year", type=str, default="2012", help="The year of dataset (2007 or 2012)")
    parser.add_argument("--data_path", type=str, default="../sharp_object_dataset/", help="the root folder of dataset")
    # parser.add_argument("--data_path", type=str, default="data/", help="the root folder of dataset")
    parser.add_argument("--pre_trained_model_type", type=str, choices=["model", "params"], default="model")
    parser.add_argument("--pre_trained_model_path", type=str, default="trained_models/whole_model_trained_yolo_voc")
    parser.add_argument("--output", type=str, default="prediction")

    args = parser.parse_args()
    return args


def test(opt):
    input_list_path = os.path.join(opt.data_path, "ImageSets/Main/{}.txt".format(opt.test_set))
    image_ids = [id.strip() for id in open(input_list_path)]
    output_folder = os.path.join(opt.output, "sharp_object_dataset_{}".format(opt.test_set))
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    if torch.cuda.is_available():
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path)
        else:
            model = Yolo(20)
            model.load_state_dict(torch.load(opt.pre_trained_model_path))
    else:
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage)
        else:
            model = Yolo(20)
            model.load_state_dict(torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage))
    model.eval()

    for id in image_ids:
        image_path = os.path.join(opt.data_path, "JPEGImages", "{}.jpg".format(id))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        image = cv2.resize(image, (opt.image_size, opt.image_size))
        image = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
        image = image[None, :, :, :]
        width_ratio = float(opt.image_size) / width
        height_ratio = float(opt.image_size) / height
        data = Variable(torch.FloatTensor(image))
        if torch.cuda.is_available():
            data = data.cuda()
        with torch.no_grad():
            logits = model(data)
            predictions = post_processing(logits, opt.image_size, CLASSES, model.anchors, opt.conf_threshold,
                                          opt.nms_threshold)
        if len(predictions) == 0:
            continue
        else:
            predictions = predictions[0]
        output_image = cv2.imread(image_path)
        bounding_boxes = []
        for pred in predictions:
            xmin = int(max(pred[0] / width_ratio, 0))
            ymin = int(max(pred[1] / height_ratio, 0))
            xmax = int(min((pred[0] + pred[2]) / width_ratio, width))
            ymax = int(min((pred[1] + pred[3]) / height_ratio, height))
            color = COLORPALETTE[CLASSES.index(pred[5])]
            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
            text_size = cv2.getTextSize(pred[5] + ' : %.2f' % pred[4], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
            cv2.putText(
                output_image, pred[5] + ' : %.2f' % pred[4],
                (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 255, 255), 1)
            print("Id: {}, Object: {}, Confidence: {}, Bounding box: ({},{}) ({},{})".format(id, pred[5], pred[4], xmin, xmax, ymin, ymax))
            bbox = str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax)
            bounding_boxes.append(
                {"class_name": pred[5], "confidence": pred[4], "file_id": id, "bbox": bbox}
            )
        with open("./input_test/detection-results/" + id + "_dr.json", "w") as outfile:
            json.dump(bounding_boxes, outfile)
        cv2.imwrite("{}/{}_prediction.jpg".format(output_folder, id), output_image)


if __name__ == "__main__":
    opt = get_args()
    test(opt)