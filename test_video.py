import argparse
import pickle
import cv2
import numpy as np
from src.utils import *
from src.yolo_net import Yolo
from src.dataset_const import *

def get_args():
    parser = argparse.ArgumentParser("You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--image_size", type=int, default=448, help="The common width and height for all images")
    # parser.add_argument("--conf_threshold", type=float, default=0.35)
    # parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--conf_threshold", type=float, default=0.1)
    parser.add_argument("--nms_threshold", type=float, default=0.1)
    parser.add_argument("--pre_trained_model_type", type=str, choices=["model", "params"], default="model")
    parser.add_argument("--pre_trained_model_path", type=str, default="trained_models/whole_model_trained_yolo_voc")
    parser.add_argument("--input", type=str, default="test_videos/input.mp4")
    parser.add_argument("--output", type=str, default="test_videos/output_voc.mp4")

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

    cap = cv2.VideoCapture(opt.input)
    out = cv2.VideoWriter(opt.output,  cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while cap.isOpened():
        flag, image = cap.read()
        output_image = np.copy(image)
        if flag:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            break
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
        if len(predictions) != 0:
            predictions = predictions[0]
            for pred in predictions:
                xmin = int(max(pred[0] / width_ratio, 0))
                ymin = int(max(pred[1] / height_ratio, 0))
                xmax = int(min((pred[0] + pred[2]) / width_ratio, width))
                ymax = int(min((pred[1] + pred[3]) / height_ratio, height))
                color = COLORPALETTE[CLASSES.index(pred[5])]
                degree_of_danger = 0
                if pred[5] == 'knife':
                    degree_of_danger = 1
                elif pred[5] == 'scissors(open)':
                    degree_of_danger = 2
                elif pred[5] == 'scissors(close)':
                    degree_of_danger = 3
                cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
                text_size = cv2.getTextSize(pred[5] + ' : %.2f' % pred[4] + ' Degree: '+ str(degree_of_danger), cv2.FONT_HERSHEY_PLAIN, 3, 1)[0]
                cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
                cv2.putText(
                    output_image, pred[5] + ' : %.2f' % pred[4] + ' Degree: '+ str(degree_of_danger),
                    (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 255, 255), 1)
        out.write(output_image)

    cap.release()
    out.release()


if __name__ == "__main__":
    opt = get_args()
    test(opt)