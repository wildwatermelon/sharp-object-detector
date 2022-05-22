import os
import glob
import argparse
import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.utils import *
from src.yolo_net import Yolo as Yolo
from src.dataset_const import *
import torch.nn as nn

def get_args():
    parser = argparse.ArgumentParser("You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--image_size", type=int, default=448, help="The common width and height for all images")
    # parser.add_argument("--conf_threshold", type=float, default=0.35)
    # parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--conf_threshold", type=float, default=0.01)
    parser.add_argument("--nms_threshold", type=float, default=0.01)
    parser.add_argument("--pre_trained_model_type", type=str, choices=["model", "params"], default="model")
    parser.add_argument("--pre_trained_model_path", type=str, default="trained_models/whole_model_trained_yolo_voc")
    parser.add_argument("--input", type=str, default="test_images_featuremap")
    parser.add_argument("--output", type=str, default="test_images")

    args = parser.parse_args()
    return args


def test(opt):
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

    for image_path in glob.iglob(opt.input + os.sep + '*.jpg'):
        print(image_path)
        if "prediction" in image_path:
            continue
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

            # start
            model_weights = []
            conv_layers = []
            model_children = list(model.children())
            counter = 0
            for i in range(len(model_children)):
                if type(model_children[i]) == nn.Conv2d:
                    counter += 1
                    model_weights.append(model_children[i].weight)
                    conv_layers.append(model_children[i])
                elif type(model_children[i]) == nn.Sequential:
                    child = model_children[i][0]
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
            print(f"Total convolution layers: {counter}")
            print("conv_layers")

            outputs = []
            names = []
            print(conv_layers)
            # set conv_layer range
            conv_layers = conv_layers[:16]

            for j in range(len(conv_layers)):
                layer = conv_layers[j]
                if len(outputs) == 0:
                    data2 = data
                else:
                    data2 = outputs[j-1]
                print(data2.shape)
                data2 = layer(data2)
                outputs.append(data2)
                names.append(str(layer))
                # print(len(outputs))
                #
                # for feature_map in outputs:
                #     print(feature_map.shape)

                processed = []
                for feature_map in outputs:
                    feature_map = feature_map.squeeze(0)
                    gray_scale = torch.sum(feature_map, 0)
                    gray_scale = gray_scale / feature_map.shape[0]
                    processed.append(gray_scale.data.cpu().numpy())
                #for fm in processed:
                #    print(fm.shape)

                fig = plt.figure(figsize=(30, 50))
                for i in range(len(processed)):
                    a = fig.add_subplot(5, 4, i + 1)
                    plt.imshow(processed[i])
                    a.axis("off")
                    a.set_title(names[i].split('(')[0], fontsize=30)
                if j == len(conv_layers)-1:
                    plt.savefig(str.format('test_feature_maps/feature_maps_{}.jpg',j), bbox_inches='tight')
            #end

if __name__ == "__main__":
    opt = get_args()
    test(opt)