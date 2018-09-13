import cv2
import torch as t
import torchvision as tv
import numpy as np
import lib.yolov2.utils as yolo_utils

class YoloV2Predict(object):
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
        sized = cv2.resize(img, (model.width, model.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        sized = t.from_numpy(sized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        sized = t.autograd.Variable(sized.cuda() if t.cuda.is_available() else sized)

        output = model(sized)
        detections = output.data

        boxes = self._get_region_boxes(
            detections, 0.5, model.num_classes, model.anchors, model.num_anchors)[0]
        boxes = self._nms(boxes, 0.4)
        result = self._get_resultbox(boxes, img)
        for left_up, right_bottom, class_name, _, prob in result:
            color = self.Color[self.VOC_CLASS.index(class_name)]
            cv2.rectangle(img,left_up,right_bottom,color,2)
            label = class_name+str(round(prob,2))
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            p1 = (left_up[0], left_up[1]- text_size[1])
            cv2.rectangle(img, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
            cv2.putText(img, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

        cv2.imwrite('%s/YOLOV2_%s_%03d.png' % (targetPath,name, epoch),img)

    def _get_region_boxes(self, output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False):
        anchor_step = int(len(anchors) / num_anchors)
        if output.dim() == 3:
            output = output.unsqueeze(0)
        batch = output.size(0)
        assert(output.size(1) == (5 + num_classes) * num_anchors)
        h = output.size(2)
        w = output.size(3)

        all_boxes = []
        output = output.view(batch * num_anchors, 5 + num_classes, h * w).transpose(
            0, 1).contiguous().view(5 + num_classes, batch * num_anchors * h * w)

        grid_x = t.linspace(0, w - 1, w).repeat(h, 1).repeat(batch * num_anchors, 1, 1).view(batch * num_anchors * h * w)
        grid_y = t.linspace(0, h - 1, h).repeat(w, 1).t().repeat(
            batch * num_anchors, 1, 1).view(batch * num_anchors * h * w)
        if t.cuda.is_available():
            grid_x = grid_x.cuda()
            grid_y = grid_y.cuda()

        xs = t.sigmoid(output[0]) + grid_x
        ys = t.sigmoid(output[1]) + grid_y

        anchor_w = t.Tensor(anchors).view(
            num_anchors, anchor_step).index_select(1, t.LongTensor([0]))
        anchor_h = t.Tensor(anchors).view(
            num_anchors, anchor_step).index_select(1, t.LongTensor([1]))
        anchor_w = anchor_w.repeat(batch, 1).repeat(
            1, 1, h * w).view(batch * num_anchors * h * w)
        anchor_h = anchor_h.repeat(batch, 1).repeat(
            1, 1, h * w).view(batch * num_anchors * h * w)
        if t.cuda.is_available():
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()

        ws = t.exp(output[2]) * anchor_w
        hs = t.exp(output[3]) * anchor_h

        det_confs = t.sigmoid(output[4])

        cls_confs = t.nn.Softmax()(
            t.autograd.Variable(output[5:5 + num_classes].transpose(0, 1))).data
        cls_max_confs, cls_max_ids = t.max(cls_confs, 1)
        cls_max_confs = cls_max_confs.view(-1)
        cls_max_ids = cls_max_ids.view(-1)

        sz_hw = h * w
        sz_hwa = sz_hw * num_anchors
        det_confs = yolo_utils.LossUtils.convert2cpu(det_confs)
        cls_max_confs = yolo_utils.LossUtils.convert2cpu(cls_max_confs)
        cls_max_ids = yolo_utils.LossUtils.convert2cpu_long(cls_max_ids)
        xs = yolo_utils.LossUtils.convert2cpu(xs)
        ys = yolo_utils.LossUtils.convert2cpu(ys)
        ws = yolo_utils.LossUtils.convert2cpu(ws)
        hs = yolo_utils.LossUtils.convert2cpu(hs)
        if validation:
            cls_confs = yolo_utils.LossUtils.convert2cpu(cls_confs.view(-1, num_classes))
        for b in range(batch):
            boxes = []
            for cy in range(h):
                for cx in range(w):
                    for i in range(num_anchors):
                        ind = b * sz_hwa + i * sz_hw + cy * w + cx
                        det_conf = det_confs[ind]
                        if only_objectness:
                            conf = det_confs[ind]
                        else:
                            conf = det_confs[ind] * cls_max_confs[ind]

                        if conf > conf_thresh:
                            bcx = xs[ind]
                            bcy = ys[ind]
                            bw = ws[ind]
                            bh = hs[ind]
                            cls_max_conf = cls_max_confs[ind]
                            cls_max_id = cls_max_ids[ind]
                            box = [bcx / w, bcy / h, bw / w, bh / h,
                                   det_conf, cls_max_conf, cls_max_id]
                            if (not only_objectness) and validation:
                                for c in range(num_classes):
                                    tmp_conf = cls_confs[ind][c]
                                    if c != cls_max_id and det_confs[ind] * tmp_conf > conf_thresh:
                                        box.append(tmp_conf)
                                        box.append(c)
                            boxes.append(box)
            all_boxes.append(boxes)
        if False:
            print('---------------------------------')
            print('matrix computation : %f' % (t1 - t0))
            print('        gpu to cpu : %f' % (t2 - t1))
            print('      boxes filter : %f' % (t3 - t2))
            print('---------------------------------')
        return all_boxes

    def _nms(self, boxes, nms_thresh):
        if len(boxes) == 0:
            return boxes

        det_confs = t.zeros(len(boxes))
        for i in range(len(boxes)):
            det_confs[i] = 1 - boxes[i][4]

        _, sortIds = t.sort(det_confs)
        out_boxes = []
        for i in range(len(boxes)):
            box_i = boxes[sortIds[i]]
            if box_i[4] > 0:
                out_boxes.append(box_i)
                for j in range(i + 1, len(boxes)):
                    box_j = boxes[sortIds[j]]
                    if yolo_utils.BoxUtils.bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                        box_j[4] = 0
        return out_boxes

    def _get_resultbox(self, detections, image):
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
                result_box.append([(x1, y1), (x2, y2), self.VOC_CLASS[cls_id], "", cls_conf])
        return result_box