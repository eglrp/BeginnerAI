import cv2
import torch as t
import torchvision as tv
import numpy as np
import lib.retinanet.utils as rn_utils

class RetinaPredict(object):
    def __init__(self, VOC_CLASS):
        self.Color = [[0, 0, 0],
                      [128, 0, 0],
                      [0, 128, 0],
                      [128, 128, 0],
                      [0, 0, 128],
                      [128, 0, 128],
                      [0, 128, 128],
                      [128, 128, 128],
                      [64, 0, 0],
                      [192, 0, 0],
                      [64, 128, 0],
                      [192, 128, 0],
                      [64, 0, 128],
                      [192, 0, 128],
                      [64, 128, 128],
                      [192, 128, 128],
                      [0, 64, 0],
                      [128, 64, 0],
                      [0, 192, 0],
                      [128, 192, 0],
                      [0, 64, 128]]
        self.VOC_CLASS = VOC_CLASS

    def predict(self, model, epoch, sourcePath, name, targetPath="outputs/"):
        img = cv2.imread(sourcePath)
        h,w,_ = img.shape

        sized = cv2.resize(img, (416, 416))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
        ])
        sized = transform(sized)
        sized = sized.unsqueeze(0)
        # sized = t.from_numpy(sized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        sized = t.autograd.Variable(sized.cuda() if t.cuda.is_available() else sized)

        loc_preds, cls_preds = model(sized)

        encoder = rn_utils.DataEncoder()
        boxes, labels, score = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (600,600))

        result = self._get_resultbox(boxes, labels, score, img)

        for left_up, right_bottom, class_name, _, prob in result:
            color = self.Color[self.VOC_CLASS.index(class_name) % 21]
            cv2.rectangle(img,left_up,right_bottom,color,2)
            label = class_name+str(round(prob,2))
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            p1 = (left_up[0], left_up[1]- text_size[1])
            cv2.rectangle(img, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
            cv2.putText(img, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

        cv2.imwrite('%s/RetinaNet_%s_%03d.png' % (targetPath, name, epoch),img)

    def _get_resultbox(self, boxes, labels, scores, image):
        result_box = []

        for box, label, score in zip(boxes, labels, scores):
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            cls_conf = float(score)
            cls_id = int(label)
            result_box.append([(x1, y1), (x2, y2), self.VOC_CLASS[cls_id], "", cls_conf])
        return result_box