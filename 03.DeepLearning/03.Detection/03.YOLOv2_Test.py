import lib.yolov2.predict as yolo_predict
import lib.yolov2.net as yolo_net
import torch as t
import os
import tqdm
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

predict = yolo_predict.YoloV2Predict((  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'))
model.eval()

if PHRASE == "Predict":

    path = os.path.join("testImages")
    listfile = os.listdir(path)
    for file in tqdm.tqdm(listfile):
        if file.endswith("jpg"):
            filename = file.split(".")[0]
            predict.predict(model, 0, os.path.join(path,"%s.jpg" % filename), filename, targetPath="results/")
else:
    pass