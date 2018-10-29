import lib.pytorch.yolov1.predict as yolo_predict
import lib.pytorch.yolov1.model as yolo_net
import lib.pytorch.detection.test as yolo_test
import torch as t
import os
import tqdm
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

PHRASE = "Predict" # "Test"
modelpath = "utils/YOLOV1.pth"
model = yolo_net.YoLoV1Net(CONFIG)
stc = t.load(modelpath) if t.cuda.is_available() else t.load(modelpath, map_location={'cuda:0': 'cpu'})
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
    path = os.path.join("../testImages")
    listfile = os.listdir(path)
    for file in tqdm.tqdm(listfile):
        if file.endswith("jpg"):
            filename = file.split(".")[0]
            predict.predict(model, 0, os.path.join(path,"%s.jpg" % filename), filename, targetPath="outputs/")
else:
    TestObj = yolo_test.Detection_Test()
    print(TestObj.calculateMAP(model, predict))