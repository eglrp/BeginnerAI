import cv2
import torch as t
import torchvision as tv
import numpy as np
import os

import lib.yolov3.utils as yolo_utils

class YoloV3Predict(object):
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

    def predict(self, model, epoch, sourcePath,name, targetPath="outputs/"):
        img = cv2.imread(sourcePath)
        h,w,_ = img.shape

        sized = cv2.resize(img, (416, 416))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        sized = t.from_numpy(sized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        sized = t.autograd.Variable(sized.cuda() if t.cuda.is_available() else sized)

        output = model(sized)
        boxes = self._get_all_boxes(output, 0.5, 20, True)[0]
        detections = self._nms(boxes, 0.4)
        if detections is not None:
            result = self._get_resultbox(detections, img, sourcePath)
            for left_up, right_bottom, class_name, _, prob in result:
                color = self.Color[self.VOC_CLASS.index(class_name) % 21]
                cv2.rectangle(img,left_up,right_bottom,color,2)
                label = class_name+str(round(prob,2))
                text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                p1 = (left_up[0], left_up[1]- text_size[1])
                cv2.rectangle(img, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
                cv2.putText(img, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

        cv2.imwrite('%s/YOLOV3_%s_%03d.png' % (targetPath, name, epoch),img)
        return img
    def _nms(self, boxes, nms_thresh):
        if len(boxes) == 0:
            return boxes

        det_confs = t.zeros(len(boxes))
        for i in range(len(boxes)):
            det_confs[i] = 1-boxes[i][4]

        _,sortIds = t.sort(det_confs)
        out_boxes = []
        for i in range(len(boxes)):
            box_i = boxes[sortIds[i]]
            if box_i[4] > 0:
                out_boxes.append(box_i)
                for j in range(i+1, len(boxes)):
                    box_j = boxes[sortIds[j]]
                    if yolo_utils.BoxUtils.bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                        #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                        box_j[4] = 0
        return out_boxes

    def _get_all_boxes(self, output, conf_thresh, num_classes, only_objectness=1, validation=False, use_cuda=True):
        # total number of inputs (batch size)
        # first element (x) for first tuple (x, anchor_mask, num_anchor)
        tot = output[0]['x'].data.size(0)
        all_boxes = [[] for i in range(tot)]
        for i in range(len(output)):
            pred, anchors, num_anchors = output[i]['x'].data, output[i]['a'], output[i]['n'].item()
            b = yolo_utils.BoxUtils.get_region_boxes(pred, conf_thresh, num_classes, anchors, num_anchors, \
                                 only_objectness=only_objectness, validation=validation, use_cuda=use_cuda)
            for t in range(tot):
                all_boxes[t] += b[t]
        return all_boxes

    def _get_resultbox(self, detections, image, imageName):
        result_box = []
        width = image.shape[1]
        height = image.shape[0]
        for box in detections:
            if len(box) >= 7:
                x1 = int(round((float(box[0].data.cpu()) - float(box[2].data.cpu()) / 2.0) * width))
                y1 = int(round((float(box[1].data.cpu()) - float(box[3].data.cpu()) / 2.0) * height))
                x2 = int(round((float(box[0].data.cpu()) + float(box[2].data.cpu()) / 2.0) * width))
                y2 = int(round((float(box[1].data.cpu()) + float(box[3].data.cpu()) / 2.0) * height))
                cls_conf = float(box[5].data.cpu())
                cls_id = int(box[6].data.cpu())
                result_box.append([(x1, y1), (x2, y2), self.VOC_CLASS[cls_id], imageName, cls_conf])
        return result_box

    def getResult(self, model, image, root):
        img = cv2.imread(os.path.join(root, image))
        h,w,_ = img.shape

        sized = cv2.resize(img, (416, 416))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        sized = t.from_numpy(sized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        sized = t.autograd.Variable(sized.cuda() if t.cuda.is_available() else sized)

        output = model(sized)
        boxes = self._get_all_boxes(output, 0.5, 20, True)[0]
        detections = self._nms(boxes, 0.4)
        result = []
        if detections is not None:
            result = self._get_resultbox(detections, img, image)

        return result
