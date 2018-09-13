'''Encode object boxes and labels.'''
import math
import torch
import random
from PIL import Image

class DataUtils(object):
    @staticmethod
    def meshgrid(x, y, row_major=True):
        a = torch.arange(0,x)
        b = torch.arange(0,y)
        xx = a.repeat(y).view(-1,1)
        yy = b.view(-1,1).repeat(1,x).view(-1,1)
        return torch.cat([xx,yy],1) if row_major else torch.cat([yy,xx],1)

class BoxUtils(object):
    @staticmethod
    def change_box_order(boxes, order):
        '''Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).

        Args:
          boxes: (tensor) bounding boxes, sized [N,4].
          order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.

        Returns:
          (tensor) converted bounding boxes, sized [N,4].
        '''
        assert order in ['xyxy2xywh','xywh2xyxy']
        a = boxes[:,:2]
        b = boxes[:,2:]
        if order == 'xyxy2xywh':
            return torch.cat([(a+b)/2,b-a+1], 1)
        return torch.cat([a-b/2,a+b/2], 1)

    @staticmethod
    def box_iou(box1, box2, order='xyxy'):
        '''Compute the intersection over union of two set of boxes.

        The default box order is (xmin, ymin, xmax, ymax).

        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
          order: (str) box order, either 'xyxy' or 'xywh'.

        Return:
          (tensor) iou, sized [N,M].

        Reference:
          https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
        '''
        if order == 'xywh':
            box1 = BoxUtils.change_box_order(box1, 'xywh2xyxy')
            box2 = BoxUtils.change_box_order(box2, 'xywh2xyxy')

        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
        rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

        wh = (rb-lt+1).clamp(min=0)      # [N,M,2]
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]+1) * (box1[:,3]-box1[:,1]+1)  # [N,]
        area2 = (box2[:,2]-box2[:,0]+1) * (box2[:,3]-box2[:,1]+1)  # [M,]
        iou = inter / (area1[:,None] + area2 - inter)
        return iou

    @staticmethod
    def box_nms(bboxes, scores, threshold=0.5, mode='union'):
        '''Non maximum suppression.

        Args:
          bboxes: (tensor) bounding boxes, sized [N,4].
          scores: (tensor) bbox scores, sized [N,].
          threshold: (float) overlap threshold.
          mode: (str) 'union' or 'min'.

        Returns:
          keep: (tensor) selected indices.

        Reference:
          https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
        '''
        x1 = bboxes[:,0]
        y1 = bboxes[:,1]
        x2 = bboxes[:,2]
        y2 = bboxes[:,3]

        areas = (x2-x1+1) * (y2-y1+1)
        _, order = scores.sort(0, descending=True)

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

            w = (xx2-xx1+1).clamp(min=0)
            h = (yy2-yy1+1).clamp(min=0)
            inter = w*h

            if mode == 'union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == 'min':
                ovr = inter / areas[order[1:]].clamp(max=areas[i])
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)

            ids = (ovr<=threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids+1]
        return torch.LongTensor(keep)
class DataEncoder:
    def __init__(self):
        self.anchor_areas = [32*32., 64*64., 128*128., 256*256., 512*512.]  # p3 -> p7
        self.aspect_ratios = [1/2., 1/1., 2/1.]
        self.scale_ratios = [1., pow(2,1/3.), pow(2,2/3.)]
        self.anchor_wh = self._get_anchor_wh()

    def _get_anchor_wh(self):
        '''Compute anchor width and height for each feature map.

        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h = ar
                h = math.sqrt(s/ar)
                w = ar * h
                for sr in self.scale_ratios:  # scale
                    anchor_h = h*sr
                    anchor_w = w*sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2)

    def _get_anchor_boxes(self, input_size):
        '''Compute anchor boxes for each feature map.

        Args:
          input_size: (tensor) model input size of (w,h).

        Returns:
          boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                        where #anchors = fmw * fmh * #anchors_per_cell
        '''
        num_fms = len(self.anchor_areas)
        fm_sizes = [(input_size/pow(2.,i+3)).ceil() for i in range(num_fms)]  # p3 -> p7 feature map sizes

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = DataUtils.meshgrid(fm_w,fm_h) + 0.5  # [fm_h*fm_w, 2]
            xy = (xy*grid_size).view(fm_h,fm_w,1,2).expand(fm_h,fm_w,9,2)
            wh = self.anchor_wh[i].view(1,1,9,2).expand(fm_h,fm_w,9,2)
            box = torch.cat([xy,wh], 3)  # [x,y,w,h]
            boxes.append(box.view(-1,4))
        return torch.cat(boxes, 0)

    def encode(self, boxes, labels, input_size):
        '''Encode target bounding boxes and class labels.

        We obey the Faster RCNN box coder:
          tx = (x - anchor_x) / anchor_w
          ty = (y - anchor_y) / anchor_h
          tw = log(w / anchor_w)
          th = log(h / anchor_h)

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        '''
        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
            else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)
        boxes = BoxUtils.change_box_order(boxes, 'xyxy2xywh')

        ious = BoxUtils.box_iou(anchor_boxes, boxes, order='xywh')
        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]

        loc_xy = (boxes[:,:2]-anchor_boxes[:,:2]) / anchor_boxes[:,2:]
        loc_wh = torch.log(boxes[:,2:]/anchor_boxes[:,2:])
        loc_targets = torch.cat([loc_xy,loc_wh], 1)
        cls_targets = 1 + labels[max_ids]

        cls_targets[max_ious<0.5] = 0
        ignore = (max_ious>0.4) & (max_ious<0.5)  # ignore ious between [0.4,0.5]
        cls_targets[ignore] = -1  # for now just mark ignored to -1
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, input_size):
        '''Decode outputs back to bouding box locations and class labels.

        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 4].
          cls_preds: (tensor) predicted class labels, sized [#anchors, #classes].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          boxes: (tensor) decode box locations, sized [#obj,4].
          labels: (tensor) class labels for each box, sized [#obj,].
        '''
        CLS_THRESH = 0.5
        NMS_THRESH = 0.5

        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
            else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)

        loc_xy = loc_preds[:,:2]
        loc_wh = loc_preds[:,2:]

        xy = loc_xy * anchor_boxes[:,2:] + anchor_boxes[:,:2]
        wh = loc_wh.exp() * anchor_boxes[:,2:]
        boxes = torch.cat([xy-wh/2, xy+wh/2], 1)  # [#anchors,4]

        score, labels = cls_preds.sigmoid().max(1)          # [#anchors,]
        ids = score > CLS_THRESH
        ids = ids.nonzero().squeeze()             # [#obj,]
        keep = BoxUtils.box_nms(boxes[ids], score[ids], threshold=NMS_THRESH)
        return boxes[ids][keep], labels[ids][keep], score[ids][keep]

class ImageUtils(object):
    @staticmethod
    def random_flip(img, boxes):
        '''Randomly flip the given PIL Image.

        Args:
            img: (PIL Image) image to be flipped.
            boxes: (tensor) object boxes, sized [#ojb,4].

        Returns:
            img: (PIL.Image) randomly flipped image.
            boxes: (tensor) randomly flipped boxes.
        '''
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = img.width
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
        return img, boxes

    @staticmethod
    def resize(img, boxes, size, max_size=1000):
        '''Resize the input PIL image to the given size.

        Args:
          img: (PIL.Image) image to be resized.
          boxes: (tensor) object boxes, sized [#ojb,4].
          size: (tuple or int)
            - if is tuple, resize image to the size.
            - if is int, resize the shorter side to the size while maintaining the aspect ratio.
          max_size: (int) when size is int, limit the image longer size to max_size.
                    This is essential to limit the usage of GPU memory.
        Returns:
          img: (PIL.Image) resized image.
          boxes: (tensor) resized boxes.
        '''
        w, h = img.size
        if isinstance(size, int):
            size_min = min(w,h)
            size_max = max(w,h)
            sw = sh = float(size) / size_min
            if sw * size_max > max_size:
                sw = sh = float(max_size) / size_max
            ow = int(w * sw + 0.5)
            oh = int(h * sh + 0.5)
        else:
            ow, oh = size
            sw = float(ow) / w
            sh = float(oh) / h
        return img.resize((ow,oh), Image.BILINEAR), \
               boxes*torch.Tensor([sw,sh,sw,sh])

    @staticmethod
    def random_crop(img, boxes):
        '''Crop the given PIL image to a random size and aspect ratio.

        A crop of random size of (0.08 to 1.0) of the original size and a random
        aspect ratio of 3/4 to 4/3 of the original aspect ratio is made.

        Args:
          img: (PIL.Image) image to be cropped.
          boxes: (tensor) object boxes, sized [#ojb,4].

        Returns:
          img: (PIL.Image) randomly cropped image.
          boxes: (tensor) randomly cropped boxes.
        '''
        success = False
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.56, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x = random.randint(0, img.size[0] - w)
                y = random.randint(0, img.size[1] - h)
                success = True
                break

        # Fallback
        if not success:
            w = h = min(img.size[0], img.size[1])
            x = (img.size[0] - w) // 2
            y = (img.size[1] - h) // 2

        img = img.crop((x, y, x+w, y+h))
        boxes -= torch.Tensor([x,y,x,y])
        boxes[:,0::2].clamp_(min=0, max=w-1)
        boxes[:,1::2].clamp_(min=0, max=h-1)
        return img, boxes

    @staticmethod
    def center_crop(img, boxes, size):
        '''Crops the given PIL Image at the center.

        Args:
          img: (PIL.Image) image to be cropped.
          boxes: (tensor) object boxes, sized [#ojb,4].
          size (tuple): desired output size of (w,h).

        Returns:
          img: (PIL.Image) center cropped image.
          boxes: (tensor) center cropped boxes.
        '''
        w, h = img.size
        ow, oh = size
        i = int(round((h - oh) / 2.))
        j = int(round((w - ow) / 2.))
        img = img.crop((j, i, j+ow, i+oh))
        boxes -= torch.Tensor([j,i,j,i])
        boxes[:,0::2].clamp_(min=0, max=ow-1)
        boxes[:,1::2].clamp_(min=0, max=oh-1)
        return img, boxes

class LossUtils(object):
    @staticmethod
    def one_hot_embedding(labels, num_classes):
        '''Embedding labels to one-hot form.

        Args:
          labels: (LongTensor) class labels, sized [N,].
          num_classes: (int) number of classes.

        Returns:
          (tensor) encoded labels, sized [N,#classes].
        '''
        y = torch.eye(num_classes)  # [D,D]
        return y[labels]            # [N,D]