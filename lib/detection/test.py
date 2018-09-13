import tqdm
import collections
import os
import numpy as np

class Detection_Test(object):
    VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                   'sofa', 'train', 'tvmonitor']

    def __init__(self, root="/input/VOC", vocTestFile="utils/voctest.txt", isContainBox=True):
        self.target = collections.defaultdict(list)
        self.image_list = []
        self.root = root
        f = open(vocTestFile)
        lines = f.readlines()
        file_list = []
        for line in lines:
            splited = line.strip().split()
            file_list.append(splited)
        f.close()
        # file_list = file_list[:100]
        if isContainBox == True:
            for image_file in tqdm.tqdm(file_list):
                image_id = image_file[0]
                self.image_list.append(image_id)
                num_obj = int(image_file[1])
                for i in range(num_obj):
                    x1 = int(image_file[2+5*i])
                    y1 = int(image_file[3+5*i])
                    x2 = int(image_file[4+5*i])
                    y2 = int(image_file[5+5*i])
                    c = int(image_file[6+5*i])
                    class_name = self.VOC_CLASSES[c]
                    self.target[(image_id,class_name)].append([x1,y1,x2,y2])
        else:
            for image_file in tqdm.tqdm(file_list):
                image_id = image_file[0]
                self.image_list.append(image_id)
                truths = np.loadtxt(image_id.replace("JPEGImages", "labels").replace("jpg", "txt"))
                truths = truths.reshape(int(truths.size / 5), 5)
                for box in truths:
                    c = int(box[0])
                    class_name = self.VOC_CLASSES[c]
                    x1 = float(box[1])
                    y1 = float(box[2])
                    w  = float(box[3])
                    h  = float(box[4])
                    x2 = float(x1 + w)
                    y2 = float(y1 + h)
                    self.target[(image_id,class_name)].append([x1,y1,x2,y2])

    def calculateMAP(self, model, predict):
        preds = collections.defaultdict(list)
        for image in tqdm.tqdm(self.image_list):
            result = predict.getResult(model, image, os.path.join(self.root, "JPEGImages"))
            for (x1, y1), (x2, y2), class_name, image_id, prob in result:  # image_id is actually image_path
                preds[class_name].append([image_id, prob, x1, y1, x2, y2])

        mAP = self._voc_eval(preds, self.target, self.VOC_CLASSES)

        return mAP

    def _voc_eval(self, preds, target, VOC_CLASSES, threshold=0.5, use_07_metric=False):
        '''
        preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
        target {(image_id,class):[[],]}

        举例：
        preds = {
            'cat': [['image01', 0.9, 20, 20, 40, 40], ['image01', 0.8, 20, 20, 50, 50], ['image02', 0.8, 30, 30, 50, 50]],
            'dog': [['image01', 0.78, 60, 60, 90, 90]]}
        target = {('image01', 'cat'): [[20, 20, 41, 41]], ('image01', 'dog'): [[60, 60, 91, 91]],
                  ('image02', 'cat'): [[30, 30, 51, 51]]}
        '''
        aps = []
        # 遍历所有的类别
        for i, class_ in enumerate(VOC_CLASSES):
            pred = preds[class_]  # [[image_id,confidence,x1,y1,x2,y2],...]
            if len(pred) == 0:  # 如果这个类别一个都没有检测到的异常情况
                ap = -1
                print('---class {} ap {}---'.format(class_, ap))
                aps += [ap]
                continue
            # print(pred)
            image_ids = [x[0] for x in pred]
            confidence = np.array([float(x[1]) for x in pred])
            BB = np.array([x[2:] for x in pred])
            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            npos = 0.
            for (key1, key2) in target:
                if key2 == class_:
                    npos += len(target[(key1, key2)])  # 统计这个类别的正样本，在这里统计才不会遗漏
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d, image_id in enumerate(image_ids):
                bb = BB[d]  # 预测框
                if (image_id, class_) in target:
                    BBGT = target[(image_id, class_)]  # [[],]
                    for bbgt in BBGT:
                        # compute overlaps
                        # intersection
                        ixmin = np.maximum(bbgt[0], bb[0])
                        iymin = np.maximum(bbgt[1], bb[1])
                        ixmax = np.minimum(bbgt[2], bb[2])
                        iymax = np.minimum(bbgt[3], bb[3])
                        iw = np.maximum(ixmax - ixmin + 1., 0.)
                        ih = np.maximum(iymax - iymin + 1., 0.)
                        inters = iw * ih

                        union = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (bbgt[2] - bbgt[0] + 1.) * (
                                bbgt[3] - bbgt[1] + 1.) - inters
                        if union == 0:
                            print(bb, bbgt)

                        overlaps = inters / union
                        if overlaps > threshold:
                            tp[d] = 1
                            BBGT.remove(bbgt)  # 这个框已经匹配到了，不能再匹配
                            if len(BBGT) == 0:
                                del target[(image_id, class_)]  # 删除没有box的键值
                            break
                    fp[d] = 1 - tp[d]
                else:
                    fp[d] = 1
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            rec = [0] if npos == float(0) else rec
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            # print(rec,prec)
            ap = self._voc_ap(rec, prec, use_07_metric)
            print('---class {} ap {}---'.format(class_, ap))
            aps += [ap]
        return np.mean(aps)

    def _voc_ap(self, rec, prec, use_07_metric=False):
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.

        else:
            # correct ap caculation
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            i = np.where(mrec[1:] != mrec[:-1])[0]

            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap