import lib.yolov2.predict as yolo_predict
import lib.yolov2.net as yolo_net
import torch as t

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names

EPOCH = 1

predict = yolo_predict.YoloV2Predict((  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'))

model = yolo_net.Darknet("utils/yolov2_voc.cfg")

if t.cuda.is_available():
    model = model.cuda()

model.load_weights("utils/yolo.weights")
# model.load_state_dict(t.load("outputs/YOLOV2_%03d.pth" % (EPOCH)))

model.eval()
predict.predict(model, EPOCH, "testImages/demo.jpg")