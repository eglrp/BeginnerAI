import lib.yolov3.predict as yolo_predict
import lib.yolov3.net as yolo_net
import torch as t

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names



predict = yolo_predict.YoloV3Predict((  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'))

model = yolo_net.Darknet("utils/yolov3.cfg", 416)

if t.cuda.is_available():
    model = model.cuda()

model.load_state_dict(t.load("outputs/YOLOV3_%03d.pth" % (1)))

model.eval()
predict.predict(model, 1, "testImages/demo.jpg")