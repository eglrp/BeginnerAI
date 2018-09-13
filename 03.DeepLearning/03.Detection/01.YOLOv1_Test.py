import lib.yolov1.predict as yolo_predict
import lib.yolov1.model as yolo_net
import lib.detection.test as yolo_test
import torch as t
import os
PHRASE = "Test" # "Test"
CONFIG = {
    "USE_GPU" : t.cuda.is_available(),
    "EPOCHS" : 120,
    "IMAGE_SIZE" : 448,
    "IMAGE_CHANNEL" : 3,
    "ALPHA" : 0.1,
    "BATCH_SIZE" : 16,
    "DATA_PATH" : "/input/VOC/JPEGImages",
    "IMAGE_LIST_FILE" : "utils/voctrain.txt",
    "CELL_NUMS" : 7,
    "CLASS_NUMS" : 20,
    "BOXES_EACH_CELL" : 2,
    "LEARNING_RATE" : 0.001,
    "L_COORD" : 5,
    "L_NOOBJ" : 0.5,
    "CELL_NUMS" : 7,
    "EACH_CELL_BOXES" : 2,
    "CLASSES" : ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor']
}

model = yolo_net.YoLoV1Net(CONFIG)
stc = t.load("results/YOLOV1.pth", map_location={'cuda:0': 'cpu'})
stc = dict(stc)
aa = dict()
for key,value in stc.items():
    if "tracked" not in key:
        aa[key] = value

model.load_state_dict(aa)

if t.cuda.is_available():
    model = model.cuda()

predict = yolo_predict.YoLoPredict(CONFIG["CLASSES"])
model.eval()

if PHRASE == "Predict":

    for file in ["demo", "dog", "eagle", "giraffe", "horses", "maskrcnn", "person"]:
        predict.predict(model, 0, os.path.join("../testImages","%s.jpg" % file), file, targetPath="results/")
else:
    TestObj = yolo_test.Detection_Test()
    print(TestObj.calculateMAP(model, predict))