import cv2
import torch as t
import torchvision as tv
import numpy as np
import os

import lib.utils.drawutils as draw

class YoLoPredict(object):
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
        image = cv2.imread(sourcePath)
        result = self._predict_gpu(model, sourcePath)
        draw.draw_box_by_cv2(image, result, '%s/yolov1_%s_%03d.png' % (targetPath, name, epoch), "voc")
        # for left_up, right_bottom, class_name, _, prob in result:
        #     color = self.Color[self.VOC_CLASS.index(class_name)]
        #     cv2.rectangle(image,left_up,right_bottom,color,2)
        #     label = class_name+str(round(prob,2))
        #     text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        #     p1 = (left_up[0], left_up[1]- text_size[1])
        #     cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        #     cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)
        #
        # cv2.imwrite('%s/yolov1_%s_%03d.png' % (targetPath, name, epoch),image)
        return image

    def _decoder(self, pred):
        grid_num = 7
        boxes=[]
        cls_indexs=[]
        probs = []
        cell_size = 1./grid_num
        pred = pred.data
        pred = pred.squeeze(0) #7x7x30
        contain1 = pred[:,:,4].unsqueeze(2)
        contain2 = pred[:,:,9].unsqueeze(2)
        contain = t.cat((contain1,contain2),2)
        mask1 = contain > 0.1 #大于阈值
        mask2 = (contain==contain.max()) #we always select the best contain_prob what ever it>0.9
        mask = (mask1+mask2).gt(0)
        # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
        for i in range(grid_num):
            for j in range(grid_num):
                for b in range(2):
                    # index = min_index[i,j]
                    # mask[i,j,index] = 0
                    if mask[i,j,b] == 1:
                        #print(i,j,b)
                        box = pred[i,j,b*5:b*5+4]
                        contain_prob = t.FloatTensor([pred[i,j,b*5+4]])
                        xy = t.FloatTensor([j,i])*cell_size #cell左上角  up left of cell
                        box[:2] = box[:2]*cell_size + xy # return cxcy relative to image
                        box_xy = t.FloatTensor(box.size())#转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                        box_xy[:2] = box[:2] - 0.5*box[2:]
                        box_xy[2:] = box[:2] + 0.5*box[2:]
                        max_prob,cls_index = t.max(pred[i,j,10:],0)
                        if float((contain_prob*max_prob)[0]) > 0.1:
                            boxes.append(box_xy.view(1,4))
                            cls_indexs.append(cls_index)
                            probs.append(contain_prob*max_prob)
        if len(boxes) ==0:
            boxes = t.zeros((1,4))
            probs = t.zeros(1)
            cls_indexs = t.zeros(1)
        else:
            boxes = t.cat(boxes,0) #(n,4)
            probs = t.cat(probs,0) #(n,)
            # cls_indexs = t.cat(cls_indexs,0) #(n,)
        keep = self._nms(boxes,probs)
        keep = keep.data.numpy()

        boxes_ = []
        cls_indexs_ = []
        probs_ = []
        for i in keep:
            boxes_.append(boxes[i])
            cls_indexs_.append(cls_indexs[i])
            probs_.append(probs[i])

        return boxes_,cls_indexs_,probs_

    def _nms(self, bboxes,scores,threshold=0.5):
        '''
        bboxes(tensor) [N,4]
        scores(tensor) [N,]
        '''
        x1 = bboxes[:,0]
        y1 = bboxes[:,1]
        x2 = bboxes[:,2]
        y2 = bboxes[:,3]
        areas = (x2-x1) * (y2-y1)

        _,order = scores.sort(0,descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)

            if order.numel() == 1:
                break

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2-xx1).clamp(min=0)
            h = (yy2-yy1).clamp(min=0)
            inter = w*h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (ovr<=threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids+1]
        return t.LongTensor(keep)

    def _predict_gpu(self, model,image_name,root_path=''):
        result = []
        image = cv2.imread(os.path.join(root_path, image_name))
        h,w,_ = image.shape
        img = cv2.resize(image,(448,448))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        mean = (123,117,104)#RGB
        img = img - np.array(mean,dtype=np.float32)

        transform = tv.transforms.Compose([tv.transforms.ToTensor(),])
        img = transform(img)
        img = t.autograd.Variable(img[None,:,:,:],volatile=True)
        if t.cuda.is_available():
            img = img.cuda()

        pred = model(img) #1x7x7x30
        pred = pred.cpu()
        boxes,cls_indexs,probs =  self._decoder(pred)

        for i,box in enumerate(boxes):
            x1 = int(box[0]*w)
            x2 = int(box[2]*w)
            y1 = int(box[1]*h)
            y2 = int(box[3]*h)
            cls_index = cls_indexs[i]
            cls_index = int(cls_index) # convert LongTensor to int
            prob = probs[i]
            prob = float(prob)
            result.append([(x1,y1),(x2,y2),self.VOC_CLASS[cls_index],prob])
        return result

    def getResult(self, model, image_name, root):
        return self._predict_gpu(model, image_name, root)