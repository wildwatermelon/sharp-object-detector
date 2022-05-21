<div align="center">   

# YOLOSO

</div>
A Modified YOLOv2 for Sharp Object Detection

### models
the models can be found in model_ref.py

### tensorboard
`tensorboard --logdir=tensorboard/yolo_voc/sharp_object_dataset`

### training
`python3 train.py --num_epoches 1000 --pre_trained_model_type model0`

### evaluation
`python test_dataset.py`
`python eval_voc.py`

## credits
yolo bootstrapped from: <br>
https://github.com/uvipen/Yolo-v2-pytorch <br>

mAP boostrapped from: <br>
https://github.com/argusswift/YOLOv4-pytorch/ <br>

labelling: <br>
https://github.com/tzutalin/labelImg <br>
