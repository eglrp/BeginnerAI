import torch
import lib.yolov3.utils as yolo_utils
import numpy as np
import math
import sys

class Darknet(torch.nn.Module):
    def getLossLayers(self):
        loss_layers = []
        for m in self.models:
            if isinstance(m, RegionLayer) or isinstance(m, YoloLayer):
                loss_layers.append(m)
        return loss_layers

    def __init__(self, cfgfile, use_cuda=True):
        super(Darknet, self).__init__()
        self.use_cuda = use_cuda
        self.blocks = yolo_utils.LoadUtils.parse_cfg(cfgfile)
        self.models = self.create_network(self.blocks) # merge conv, bn,leaky
        self.loss_layers = self.getLossLayers()

        if len(self.loss_layers) > 0:
            last = len(self.loss_layers)-1
            self.anchors = self.loss_layers[last].anchors
            self.num_anchors = self.loss_layers[last].num_anchors
            self.anchor_step = self.loss_layers[last].anchor_step
            self.num_classes = self.loss_layers[last].num_classes

        # default format : major=0, minor=1
        self.header = torch.IntTensor([0,1,0,0])
        self.seen = 0

    def forward(self, x):
        ind = -2
        self.loss_layers = None
        outputs = dict()
        out_boxes = dict()
        outno = 0
        for block in self.blocks:
            ind = ind + 1

            if block['type'] == 'net':
                continue
            elif block['type'] in ['convolutional', 'maxpool', 'reorg', 'upsample', 'avgpool', 'softmax', 'connected']:
                if self.use_cuda:
                    x = x.cuda()
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1,x2),1)
                outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind-1]
                x  = x1 + x2
                if activation == 'leaky':
                    x = torch.nn.functional.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = torch.nn.functional.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] in [ 'region', 'yolo']:
                boxes = self.models[ind].get_mask_boxes(x)
                out_boxes[outno]= boxes
                outno += 1
                outputs[ind] = None
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))
        return x if outno == 0 else out_boxes

    def print_network(self):
        yolo_utils.LoadUtils.print_cfg(self.blocks)

    def create_network(self, blocks):
        models = torch.nn.ModuleList()

        prev_filters = 3
        out_filters =[]
        prev_stride = 1
        out_strides = []
        conv_id = 0
        ind = -2
        for block in blocks:
            ind += 1
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                self.width = int(block['width'])
                self.height = int(block['height'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size-1)//2 if is_pad else 0
                activation = block['activation']
                model = torch.nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), torch.nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), torch.nn.BatchNorm2d(filters))
                    #model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), torch.nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), torch.nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), torch.nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = torch.nn.MaxPool2d(pool_size, stride)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = torch.nn.Softmax()
                out_strides.append(prev_stride)
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = torch.nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = torch.nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = torch.nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                prev_stride = prev_stride * stride
                out_strides.append(prev_stride)
                models.append(Reorg(stride))
            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                out_filters.append(prev_filters)
                prev_stride = prev_stride / stride
                out_strides.append(prev_stride)
                #models.append(nn.Upsample(scale_factor=stride, mode='nearest'))
                models.append(Upsample(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                    prev_stride = out_strides[layers[0]]
                elif len(layers) == 2:
                    assert(layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                    prev_stride = out_strides[layers[0]]
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind-1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind-1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = torch.nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = torch.nn.Sequential(
                        torch.nn.Linear(prev_filters, filters),
                        torch.nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = torch.nn.Sequential(
                        torch.nn.Linear(prev_filters, filters),
                        torch.nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'region':
                region_layer = RegionLayer(use_cuda=self.use_cuda)
                anchors = block['anchors'].split(',')
                region_layer.anchors = [float(i) for i in anchors]
                region_layer.num_classes = int(block['classes'])
                region_layer.num_anchors = int(block['num'])
                region_layer.anchor_step = len(region_layer.anchors)//region_layer.num_anchors
                region_layer.rescore = int(block['rescore'])
                region_layer.object_scale = float(block['object_scale'])
                region_layer.noobject_scale = float(block['noobject_scale'])
                region_layer.class_scale = float(block['class_scale'])
                region_layer.coord_scale = float(block['coord_scale'])
                region_layer.thresh = float(block['thresh'])
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(region_layer)
            elif block['type'] == 'yolo':
                yolo_layer = YoloLayer(use_cuda=self.use_cuda)
                anchors = block['anchors'].split(',')
                anchor_mask = block['mask'].split(',')
                yolo_layer.anchor_mask = [int(i) for i in anchor_mask]
                yolo_layer.anchors = [float(i) for i in anchors]
                yolo_layer.num_classes = int(block['classes'])
                yolo_layer.num_anchors = int(block['num'])
                yolo_layer.anchor_step = len(yolo_layer.anchors)//yolo_layer.num_anchors
                try:
                    yolo_layer.rescore = int(block['rescore'])
                except:
                    pass
                yolo_layer.ignore_thresh = float(block['ignore_thresh'])
                yolo_layer.truth_thresh = float(block['truth_thresh'])
                yolo_layer.stride = prev_stride
                yolo_layer.nth_layer = ind
                yolo_layer.net_width = self.width
                yolo_layer.net_height = self.height
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(yolo_layer)
            else:
                print('unknown type %s' % (block['type']))

        return models

    def load_binfile(self, weightfile):
        fp = open(weightfile, 'rb')

        version = np.fromfile(fp, count=3, dtype=np.int32)
        version = [int(i) for i in version]
        if version[0]*10+version[1] >=2 and version[0] < 1000 and version[1] < 1000:
            seen = np.fromfile(fp, count=1, dtype=np.int64)
        else:
            seen = np.fromfile(fp, count=1, dtype=np.int32)
        self.header = torch.from_numpy(np.concatenate((version, seen), axis=0))
        self.seen = int(seen)
        body = np.fromfile(fp, dtype=np.float32)
        fp.close()
        return body

    def load_weights(self, weightfile):
        buf = self.load_binfile(weightfile)

        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = yolo_utils.LoadUtils.load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = yolo_utils.LoadUtils.load_conv(buf, start, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = yolo_utils.LoadUtils.load_fc(buf, start, model[0])
                else:
                    start = yolo_utils.LoadUtils.load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'yolo':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))

    def save_weights(self, outfile, cutoff=0):
        if cutoff <= 0:
            cutoff = len(self.blocks)-1

        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = np.array(self.header[0:3].numpy(), np.int32)
        header.tofile(fp)
        if (self.header[0]*10+self.header[1]) >= 2:
            seen = np.array(self.seen, np.int64)
        else:
            seen = np.array(self.seen, np.int32)
        seen.tofile(fp)

        ind = -1
        for blockId in range(1, cutoff+1):
            ind = ind + 1
            block = self.blocks[blockId]
            if block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    yolo_utils.LoadUtils.save_conv_bn(fp, model[0], model[1])
                else:
                    yolo_utils.LoadUtils.save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    yolo_utils.LoadUtils.save_fc(fc, model)
                else:
                    yolo_utils.LoadUtils.save_fc(fc, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'yolo':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))
        fp.close()

class RegionLayer(torch.nn.Module):
    def __init__(self, num_classes=0, anchors=[1.0], num_anchors=1, use_cuda=None):
        super(RegionLayer, self).__init__()
        use_cuda = torch.cuda.is_available() and (True if use_cuda is None else use_cuda)
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)//num_anchors
        #self.anchors = torch.stack(torch.cuda.FloatTensor(anchors).split(self.anchor_step)).to(self.device)
        self.anchors = torch.cuda.FloatTensor(anchors).view(self.num_anchors, self.anchor_step).to(self.device)
        self.rescore = 1
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0

    def build_targets(self, pred_boxes, target, nH, nW):
        nB = target.size(0)
        nA = self.num_anchors
        conf_mask  = torch.ones (nB, nA, nH, nW) * self.noobject_scale
        coord_mask = torch.zeros(nB, nA, nH, nW)
        cls_mask   = torch.zeros(nB, nA, nH, nW)
        tcoord     = torch.zeros( 4, nB, nA, nH, nW)
        tconf      = torch.zeros(nB, nA, nH, nW)
        tcls       = torch.zeros(nB, nA, nH, nW)

        nAnchors = nA*nH*nW
        nPixels  = nH*nW
        nGT = 0 # number of ground truth
        nRecall = 0
        # it works faster on CPU than on GPU.
        anchors = self.anchors.to("cpu")

        if self.seen < 12800:
            tcoord[0].fill_(0.5)
            tcoord[1].fill_(0.5)
            coord_mask.fill_(1)

        for b in range(nB):
            cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors].t()
            cur_ious = torch.zeros(nAnchors)
            tbox = target[b].view(-1,5).to("cpu")
            for t in range(50):
                if tbox[t][1] == 0:
                    break
                gx, gw = [ i * nW for i in (tbox[t][1], tbox[t][3]) ]
                gy, gh = [ i * nH for i in (tbox[t][2], tbox[t][4]) ]
                cur_gt_boxes = torch.cuda.FloatTensor([gx, gy, gw, gh]).repeat(nAnchors,1).t()
                cur_ious = torch.max(cur_ious, yolo_utils.BoxUtils.multi_bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
            ignore_ix = cur_ious>self.thresh
            conf_mask[b][ignore_ix.view(nA,nH,nW)] = 0

            for t in range(50):
                if tbox[t][1] == 0:
                    break
                nGT += 1
                gx, gw = [ i * nW for i in (tbox[t][1], tbox[t][3]) ]
                gy, gh = [ i * nH for i in (tbox[t][2], tbox[t][4]) ]
                gw, gh = gw.float(), gh.float()
                gi, gj = int(gx), int(gy)

                tmp_gt_boxes = torch.cuda.FloatTensor([0, 0, gw, gh]).repeat(nA,1).t()
                anchor_boxes = torch.cat((torch.zeros(nA, 2), anchors),1).t()
                tmp_ious = yolo_utils.BoxUtils.multi_bbox_ious(tmp_gt_boxes, anchor_boxes, x1y1x2y2=False)
                best_iou, best_n = torch.max(tmp_ious, 0)

                if self.anchor_step == 4: # this part is not tested.
                    tmp_ious_mask = (tmp_ious==best_iou)
                    if tmp_ious_mask.sum() > 0:
                        gt_pos = torch.cuda.FloatTensor([gi, gj, gx, gy]).repeat(nA,1).t()
                        an_pos = anchor_boxes[4:6] # anchor_boxes are consisted of [0 0 aw ah ax ay]
                        dist = pow(((gt_pos[0]+an_pos[0])-gt_pos[2]),2) + pow(((gt_pos[1]+an_pos[1])-gt_pos[3]),2)
                        dist[1-tmp_ious_mask]=10000 # set the large number for the small ious
                        _, best_n = torch.min(dist,0)

                gt_box = torch.cuda.FloatTensor([gx, gy, gw, gh])
                pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]
                iou = yolo_utils.BoxUtils.bbox_iou(gt_box, pred_box, x1y1x2y2=False)

                coord_mask[b][best_n][gj][gi] = 1
                cls_mask  [b][best_n][gj][gi] = 1
                conf_mask [b][best_n][gj][gi] = self.object_scale
                tcoord [0][b][best_n][gj][gi] = gx - gi
                tcoord [1][b][best_n][gj][gi] = gy - gj
                tcoord [2][b][best_n][gj][gi] = math.log(gw/anchors[best_n][0])
                tcoord [3][b][best_n][gj][gi] = math.log(gh/anchors[best_n][1])
                tcls      [b][best_n][gj][gi] = tbox[t][0]
                tconf     [b][best_n][gj][gi] = iou if self.rescore else 1.
                if iou > 0.5:
                    nRecall += 1

        return nGT, nRecall, coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls

    def get_mask_boxes(self, output):
        if not isinstance(self.anchors, torch.Tensor):
            self.anchors = torch.cuda.FloatTensor(self.anchors).view(self.num_anchors, self.anchor_step).to(self.device)
        masked_anchors = self.anchors.view(-1)
        num_anchors = torch.cuda.IntTensor([self.num_anchors]).to(self.device)
        return {'x':output, 'a':masked_anchors, 'n':num_anchors}

    def forward(self, output, target):
        #output : BxAs*(4+1+num_classes)*H*W
        nB = output.data.size(0)    # batch size
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        cls_anchor_dim = nB*nA*nH*nW

        if not isinstance(self.anchors, torch.Tensor):
            self.anchors = torch.cuda.FloatTensor(self.anchors).view(self.num_anchors, self.anchor_step).to(self.device)

        output = output.view(nB, nA, (5+nC), nH, nW)
        cls_grid = torch.linspace(5,5+nC-1,nC).long().to(self.device)
        ix = torch.LongTensor(range(0,5)).to(self.device)
        pred_boxes = torch.cuda.FloatTensor(4, cls_anchor_dim).to(self.device)

        coord = output.index_select(2, ix[0:4]).view(nB*nA, -1, nH*nW).transpose(0,1).contiguous().view(-1,cls_anchor_dim)  # x, y, w, h
        coord[0:2] = coord[0:2].sigmoid()                                   # x, y
        conf = output.index_select(2, ix[4]).view(nB, nA, nH, nW).sigmoid()
        cls  = output.index_select(2, cls_grid)
        cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(cls_anchor_dim, nC)

        grid_x = torch.linspace(0, nW-1, nW).repeat(nB*nA, nH, 1).view(cls_anchor_dim).to(self.device)
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(cls_anchor_dim).to(self.device)
        anchor_w = self.anchors.index_select(1, ix[0]).repeat(1, nB*nH*nW).view(cls_anchor_dim)
        anchor_h = self.anchors.index_select(1, ix[1]).repeat(1, nB*nH*nW).view(cls_anchor_dim)

        pred_boxes[0] = coord[0] + grid_x
        pred_boxes[1] = coord[1] + grid_y
        pred_boxes[2] = coord[2].exp() * anchor_w
        pred_boxes[3] = coord[3].exp() * anchor_h
        # for build_targets. it works faster on CPU than on GPU
        pred_boxes = yolo_utils.LoadUtils.convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4)).detach()

        nGT, nRecall, coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = \
            self.build_targets(pred_boxes, target.detach(), nH, nW)

        cls_mask = (cls_mask == 1)
        tcls = tcls[cls_mask].long().view(-1)
        cls_mask = cls_mask.view(-1, 1).repeat(1,nC).to(self.device)
        cls = cls[cls_mask].view(-1, nC)

        nProposals = int((conf > 0.25).sum())

        tcoord = tcoord.view(4, cls_anchor_dim).to(self.device)
        tconf, tcls = tconf.to(self.device), tcls.to(self.device)
        coord_mask, conf_mask = coord_mask.view(cls_anchor_dim).to(self.device), conf_mask.sqrt().to(self.device)

        loss_coord = self.coord_scale * torch.nn.MSELoss(size_average=False)(coord*coord_mask, tcoord*coord_mask)/2
        # sqrt(object_scale)/2 is almost equal to 1.
        loss_conf = torch.nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2
        loss_cls = self.class_scale * torch.nn.CrossEntropyLoss(size_average=False)(cls, tcls) if cls.size(0) > 0 else 0
        loss = loss_coord + loss_conf + loss_cls

        if math.isnan(loss.item()):
            print(conf, tconf)
            sys.exit(0)
        return loss

class YoloLayer(torch.nn.Module):
    def __init__(self, anchor_mask=[], num_classes=0, anchors=[1.0], num_anchors=1, use_cuda=None):
        super(YoloLayer, self).__init__()
        use_cuda = torch.cuda.is_available() and (True if use_cuda is None else use_cuda)
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors)//num_anchors
        self.rescore = 0
        self.ignore_thresh = 0.5
        self.truth_thresh = 1.
        self.stride = 32
        self.nth_layer = 0
        self.seen = 0
        self.net_width = 0
        self.net_height = 0

    def get_mask_boxes(self, output):
        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m*self.anchor_step:(m+1)*self.anchor_step]
        masked_anchors = [anchor/self.stride for anchor in masked_anchors]

        masked_anchors = torch.cuda.FloatTensor(masked_anchors).to(self.device) if torch.cuda.is_available() else torch.FloatTensor(masked_anchors)
        num_anchors = torch.IntTensor([len(self.anchor_mask)]).to(self.device)
        return {'x':output, 'a':masked_anchors, 'n':num_anchors}

    def build_targets(self, pred_boxes, target, anchors, nA, nH, nW):
        nB = target.size(0)
        anchor_step = anchors.size(1) # anchors[nA][anchor_step]
        conf_mask  = torch.ones (nB, nA, nH, nW)
        coord_mask = torch.zeros(nB, nA, nH, nW)
        cls_mask   = torch.zeros(nB, nA, nH, nW)
        tcoord     = torch.zeros( 4, nB, nA, nH, nW)
        tconf      = torch.zeros(nB, nA, nH, nW)
        tcls       = torch.zeros(nB, nA, nH, nW)
        twidth, theight = self.net_width/self.stride, self.net_height/self.stride

        nAnchors = nA*nH*nW
        nPixels  = nH*nW
        nGT = 0
        nRecall = 0
        nRecall75 = 0

        # it works faster on CPU than on GPU.
        anchors = anchors.to("cpu")

        for b in range(nB):
            cur_pred_boxes = pred_boxes[b*nAnchors:(b+1)*nAnchors].t()
            cur_ious = torch.zeros(nAnchors)
            tbox = target[b].view(-1,5).to("cpu")
            for t in range(50):
                if tbox[t][1] == 0:
                    break
                gx, gy = tbox[t][1] * nW, tbox[t][2] * nH
                gw, gh = tbox[t][3] * twidth, tbox[t][4] * theight
                cur_gt_boxes = torch.cuda.FloatTensor([gx, gy, gw, gh]).repeat(nAnchors,1).t()
                cur_ious = torch.max(cur_ious, yolo_utils.BoxUtils.multi_bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
            ignore_ix = cur_ious>self.ignore_thresh
            conf_mask[b][ignore_ix.view(nA,nH,nW)] = 0

            for t in range(50):
                if tbox[t][1] == 0:
                    break
                nGT += 1
                gx, gy = tbox[t][1] * nW, tbox[t][2] * nH
                gw, gh = tbox[t][3] * twidth, tbox[t][4] * theight
                gw, gh = gw.float(), gh.float()
                gi, gj = int(gx), int(gy)

                tmp_gt_boxes = torch.cuda.FloatTensor([0, 0, gw, gh]).repeat(nA,1).t()
                anchor_boxes = torch.cat((torch.zeros(nA, anchor_step), anchors.cuda()),1).t()
                _, best_n = torch.max(yolo_utils.BoxUtils.multi_bbox_ious(tmp_gt_boxes, anchor_boxes, x1y1x2y2=False), 0)

                gt_box = torch.cuda.FloatTensor([gx, gy, gw, gh])
                pred_box = pred_boxes[b*nAnchors+best_n*nPixels+gj*nW+gi]
                iou = yolo_utils.BoxUtils.bbox_iou(gt_box, pred_box, x1y1x2y2=False)

                coord_mask[b][best_n][gj][gi] = 1
                cls_mask  [b][best_n][gj][gi] = 1
                conf_mask [b][best_n][gj][gi] = 1
                tcoord [0][b][best_n][gj][gi] = gx - gi
                tcoord [1][b][best_n][gj][gi] = gy - gj
                tcoord [2][b][best_n][gj][gi] = math.log(gw/anchors[best_n][0])
                tcoord [3][b][best_n][gj][gi] = math.log(gh/anchors[best_n][1])
                tcls      [b][best_n][gj][gi] = tbox[t][0]
                tconf     [b][best_n][gj][gi] = iou if self.rescore else 1.

                if iou > 0.5:
                    nRecall += 1
                    if iou > 0.75:
                        nRecall75 += 1

        return nGT, nRecall, nRecall75, coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls

    def forward(self, output, target):
        #output : BxAs*(4+1+num_classes)*H*W
        mask_tuple = self.get_mask_boxes(output)
        nB = output.data.size(0)    # batch size
        nA = mask_tuple['n'].item() # num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)
        anchor_step = mask_tuple['a'].size(0)//nA
        anchors = mask_tuple['a'].view(nA, anchor_step).to(self.device)
        cls_anchor_dim = nB*nA*nH*nW

        # print(output.shape)
        # print(nB, nA, nC, nH, nW)

        output  = output.view(nB, nA, (5+nC), nH, nW)
        cls_grid = torch.linspace(5,5+nC-1,nC).long().to(self.device)
        ix = torch.LongTensor(range(0,5)).to(self.device)
        pred_boxes = torch.cuda.FloatTensor(4, cls_anchor_dim).to(self.device)

        coord = output.index_select(2, ix[0:4]).view(nB*nA, -1, nH*nW).transpose(0,1).contiguous().view(-1,cls_anchor_dim)  # x, y, w, h
        coord[0:2] = coord[0:2].sigmoid()                                   # x, y
        conf = output.index_select(2, ix[4]).view(nB, nA, nH, nW).sigmoid()
        cls  = output.index_select(2, cls_grid)
        cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(cls_anchor_dim, nC)

        grid_x = torch.linspace(0, nW-1, nW).repeat(nB*nA, nH, 1).view(cls_anchor_dim).to(self.device)
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(cls_anchor_dim).to(self.device)
        anchor_w = anchors.index_select(1, ix[0]).repeat(1, nB*nH*nW).view(cls_anchor_dim)
        anchor_h = anchors.index_select(1, ix[1]).repeat(1, nB*nH*nW).view(cls_anchor_dim)

        pred_boxes[0] = coord[0] + grid_x
        pred_boxes[1] = coord[1] + grid_y
        pred_boxes[2] = coord[2].exp() * anchor_w
        pred_boxes[3] = coord[3].exp() * anchor_h
        # for build_targets. it works faster on CPU than on GPU
        pred_boxes = yolo_utils.LoadUtils.convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,4)).detach()

        nGT, nRecall, nRecall75, coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = \
            self.build_targets(pred_boxes, target.detach(), anchors.detach(), nA, nH, nW)

        cls_mask = (cls_mask == 1)
        tcls = tcls[cls_mask].long().view(-1)
        cls_mask = cls_mask.view(-1, 1).repeat(1,nC).to(self.device)
        cls = cls[cls_mask].view(-1, nC)

        nProposals = int((conf > 0.25).sum())

        tcoord = tcoord.view(4, cls_anchor_dim).to(self.device)
        tconf, tcls = tconf.to(self.device), tcls.to(self.device)
        coord_mask, conf_mask = coord_mask.view(cls_anchor_dim).to(self.device), conf_mask.to(self.device)

        loss_coord = torch.nn.MSELoss(size_average=False)(coord*coord_mask, tcoord*coord_mask)/2
        loss_conf = torch.nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)
        loss_cls = torch.nn.CrossEntropyLoss(size_average=False)(cls, tcls) if cls.size(0) > 0 else 0
        loss = loss_coord + loss_conf + loss_cls


        if math.isnan(loss.item()):
            print(conf, tconf)
            sys.exit(0)
        return loss

class MaxPoolStride1(torch.nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = torch.nn.functional.max_pool2d(torch.nn.functional.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class Upsample(torch.nn.Module):
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, hs, W, ws).contiguous().view(B, C, H*hs, W*ws)
        return x

class Reorg(torch.nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H//hs, hs, W//ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, (H//hs)*(W//ws), hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H//hs, W//ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H//hs, W//ws)
        return x

class GlobalAvgPool2d(torch.nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = torch.nn.functional.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x

# for route and shortcut
class EmptyModule(torch.nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x
