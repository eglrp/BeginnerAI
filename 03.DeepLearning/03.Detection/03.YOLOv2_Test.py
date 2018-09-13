import lib.yolov2.predict as yolo_predict
import lib.yolov2.net as yolo_net
import torch as t
import os
PHRASE = "Predict" # "Test"

model = yolo_net.Darknet("utils/yolov2_voc.cfg")
stc = t.load("results/YOLOV2.pth", map_location={'cuda:0': 'cpu'})
stc = dict(stc)
aa = dict()
for key,value in stc.items():
    if "tracked" not in key:
        aa[key] = value

model.load_state_dict(aa)

if t.cuda.is_available():
    model = model.cuda()

if PHRASE == "Predict":
    predict = yolo_predict.YoloV2Predict((  # always index 0
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'))
    model.eval()
    for file in ["demo", "dog", "eagle", "giraffe", "horses", "maskrcnn", "person"]:
        predict.predict(model, 0, os.path.join("../testImages","%s.jpg" % file), file, targetPath="results/")
else:
    pass