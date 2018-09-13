import lib.yolov3.predict as yolo_predict
import lib.yolov3.net as yolo_net
import lib.detection.test as yolo_test
import torch as t
import os
import tqdm

PHRASE = "Predict" # "Test"

modelpath = "results/YOLOV3.pth"
model = yolo_net.Darknet("utils/yolo_v3.cfg", use_cuda=t.cuda.is_available())
stc = t.load(modelpath, map_location={'cuda:0': 'cpu'}) if not t.cuda.is_available() else t.load(modelpath)
stc = dict(stc)
aa = dict()
for key,value in stc.items():
    if "tracked" not in key:
        aa[key] = value

model.load_state_dict(aa)

if t.cuda.is_available():
    model = model.cuda()

predict = yolo_predict.YoloV3Predict((  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'))
model.eval()

if PHRASE == "Predict":
    path = os.path.join("../testImages")
    listfile = os.listdir(path)
    for file in tqdm.tqdm(listfile):
        if file.endswith("jpg"):
            filename = file.split(".")[0]
            predict.predict(model, 0, os.path.join(path,"%s.jpg" % filename), filename, targetPath="results/")
else:
    TestObj = yolo_test.Detection_Test(vocTestFile="utils/voctest.txt")
    print(TestObj.calculateMAP(model, predict))