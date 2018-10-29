import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import lib.pytorch.yolov2.utils as yolo_utils

class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = int(len(anchors) / num_anchors)
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

    def forward(self, output, target):
        # output : BxAs*(4+1+num_classes)*H*W
        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        output = output.view(nB, nA, (5 + nC), nH, nW)
        x = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]) if torch.cuda.is_available() else torch.LongTensor([0]))).view(nB, nA, nH, nW))
        y = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]) if torch.cuda.is_available() else torch.LongTensor([1]))).view(nB, nA, nH, nW))
        w = output.index_select(2, Variable(torch.cuda.LongTensor([2]) if torch.cuda.is_available() else torch.LongTensor([2]))).view(nB, nA, nH, nW)
        h = output.index_select(2, Variable(torch.cuda.LongTensor([3]) if torch.cuda.is_available() else torch.LongTensor([3]))).view(nB, nA, nH, nW)
        conf = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([4]) if torch.cuda.is_available() else torch.LongTensor([4]))).view(nB, nA, nH, nW))
        cls = output.index_select(2, Variable(torch.linspace(5, 5 + nC - 1, nC).long().cuda() if torch.cuda.is_available() else torch.linspace(5, 5 + nC - 1, nC).long()))
        cls = cls.view(nB * nA, nC, nH * nW).transpose(1,2).contiguous().view(nB * nA * nH * nW, nC)
        t1 = time.time()

        pred_boxes = torch.FloatTensor(4, nB * nA * nH * nW)
        grid_x = torch.linspace(
            0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA, 1, 1).view(nB * nA * nH * nW)
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(nB * nA * nH * nW)

        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0]))

        anchor_h = torch.Tensor(self.anchors).view(
            nA, self.anchor_step).index_select(1, torch.LongTensor([1]))
        anchor_w = anchor_w.repeat(nB, 1).repeat(
            1, 1, nH * nW).view(nB * nA * nH * nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(
            1, 1, nH * nW).view(nB * nA * nH * nW)

        if torch.cuda.is_available():
            grid_x = grid_x.cuda()
            grid_y = grid_y.cuda()
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()

        pred_boxes[0] = x.data.view(grid_x.shape) + grid_x
        pred_boxes[1] = y.data.view(grid_y.shape) + grid_y
        pred_boxes[2] = torch.exp(w.data).view(anchor_w.shape) * anchor_w
        pred_boxes[3] = torch.exp(h.data).view(anchor_h.shape) * anchor_h
        pred_boxes = yolo_utils.LossUtils.convert2cpu(
            pred_boxes.transpose(0, 1).contiguous().view(-1, 4))
        t2 = time.time()

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls = yolo_utils.LossUtils.build_targets(pred_boxes, target.data, self.anchors, nA, nC,
                                                                                                    nH, nW, self.noobject_scale, self.object_scale, self.thresh, self.seen)
        cls_mask = (cls_mask == 1)
        nProposals = int((conf > 0.25).sum().data[0])

        tx = Variable(tx)
        ty = Variable(ty)
        tw = Variable(tw)
        th = Variable(th)
        tconf = Variable(tconf)

        tcls_temp = tcls[cls_mask]
        tcls = Variable(tcls_temp.long())

        coord_mask = Variable(coord_mask)
        conf_mask = Variable(conf_mask.sqrt())
        cls_mask = Variable(cls_mask.view(-1, 1).repeat(1, nC))

        if torch.cuda.is_available():
            coord_mask = coord_mask.cuda()
            conf_mask = conf_mask.cuda()
            cls_mask = cls_mask.cuda()
            tx = tx.cuda()
            ty = ty.cuda()
            tw = tw.cuda()
            th = th.cuda()
            tconf = tconf.cuda()
            tcls = tcls.cuda()
            # self.coord_scale = self.coord_scale.cuda()

        cls = cls[cls_mask].view(-1, nC)

        t3 = time.time()

        loss_x = self.coord_scale * nn.MSELoss(size_average=False)(x * coord_mask, tx * coord_mask) / 2.0
        loss_y = self.coord_scale * nn.MSELoss(size_average=False)(y * coord_mask, ty * coord_mask) / 2.0
        loss_w = self.coord_scale * nn.MSELoss(size_average=False)(w * coord_mask, tw * coord_mask) / 2.0
        loss_h = self.coord_scale * nn.MSELoss(size_average=False)(h * coord_mask, th * coord_mask) / 2.0
        loss_conf = nn.MSELoss(size_average=False)(conf * conf_mask, tconf * conf_mask) / 2.0

        loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)

        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
        t4 = time.time()
        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        # print('%d: nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (self.seen, nGT, nCorrect,nProposals, loss_x.data[0], loss_y.data[0], loss_w.data[0], loss_h.data[0], loss_conf.data[0], loss_cls.data[0], loss.data[0]))
        return loss