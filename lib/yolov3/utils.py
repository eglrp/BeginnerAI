import torch

import random
import os
from PIL import Image
import numpy as np

class LoadUtils(object):
    @staticmethod
    def read_data_cfg(datacfg):
        options = dict()
        options['gpus'] = '0'
        options['num_workers'] = '0'
        with open(datacfg, 'r') as fp:
            lines = fp.readlines()

        for line in lines:
            line = line.strip()
            if line == '':
                continue
            key,value = line.split('=')
            key = key.strip()
            value = value.strip()
            options[key] = value
        return options

    @staticmethod
    def parse_cfg(cfgfile):
        blocks = []
        fp = open(cfgfile, 'r')
        block =  None
        line = fp.readline()
        while line != '':
            line = line.rstrip()
            if line == '' or line[0] == '#':
                line = fp.readline()
                continue
            elif line[0] == '[':
                if block:
                    blocks.append(block)
                block = dict()
                block['type'] = line.lstrip('[').rstrip(']')
                # set default value
                if block['type'] == 'convolutional':
                    block['batch_normalize'] = 0
            else:
                key,value = line.split('=')
                key = key.strip()
                if key == 'type':
                    key = '_type'
                value = value.strip()
                block[key] = value
            line = fp.readline()

        if block:
            blocks.append(block)
        fp.close()
        return blocks

    @staticmethod
    def file_lines(thefilepath):
        count = 0
        thefile = open(thefilepath, 'rb')
        while True:
            buffer = thefile.read(8192*1024)
            if not buffer:
                break
            count += buffer.count(b'\n')
        thefile.close( )
        return count

    @staticmethod
    def print_cfg(blocks):
        print('layer     filters    size              input                output');
        prev_width = 416
        prev_height = 416
        prev_filters = 3
        out_filters =[]
        out_widths =[]
        out_heights =[]
        ind = -2
        for block in blocks:
            ind = ind + 1
            if block['type'] == 'net':
                prev_width = int(block['width'])
                prev_height = int(block['height'])
                continue
            elif block['type'] == 'convolutional':
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size-1)//2 if is_pad else 0
                width = (prev_width + 2*pad - kernel_size)//stride + 1
                height = (prev_height + 2*pad - kernel_size)//stride + 1
                print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'conv', filters, kernel_size, kernel_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
                prev_width = width
                prev_height = height
                prev_filters = filters
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                width = prev_width//stride
                height = prev_height//stride
                print('%5d %-6s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'max', pool_size, pool_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
                prev_width = width
                prev_height = height
                prev_filters = filters
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'avgpool':
                width = 1
                height = 1
                print('%5d %-6s                   %3d x %3d x%4d   ->  %3d' % (ind, 'avg', prev_width, prev_height, prev_filters,  prev_filters))
                prev_width = width
                prev_height = height
                prev_filters = filters
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'softmax':
                print('%5d %-6s                                    ->  %3d' % (ind, 'softmax', prev_filters))
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'cost':
                print('%5d %-6s                                     ->  %3d' % (ind, 'cost', prev_filters))
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                filters = stride * stride * prev_filters
                width = prev_width//stride
                height = prev_height//stride
                print('%5d %-6s             / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'reorg', stride, prev_width, prev_height, prev_filters, width, height, filters))
                prev_width = width
                prev_height = height
                prev_filters = filters
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                filters = prev_filters
                width = prev_width*stride
                height = prev_height*stride
                print('%5d %-6s           * %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'upsample', stride, prev_width, prev_height, prev_filters, width, height, filters))
                prev_width = width
                prev_height = height
                prev_filters = filters
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    print('%5d %-6s %d' % (ind, 'route', layers[0]))
                    prev_width = out_widths[layers[0]]
                    prev_height = out_heights[layers[0]]
                    prev_filters = out_filters[layers[0]]
                elif len(layers) == 2:
                    print('%5d %-6s %d %d' % (ind, 'route', layers[0], layers[1]))
                    prev_width = out_widths[layers[0]]
                    prev_height = out_heights[layers[0]]
                    assert(prev_width == out_widths[layers[1]])
                    assert(prev_height == out_heights[layers[1]])
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] in ['region', 'yolo']:
                print('%5d %-6s' % (ind, 'detection'))
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'shortcut':
                from_id = int(block['from'])
                from_id = from_id if from_id > 0 else from_id+ind
                print('%5d %-6s %d' % (ind, 'shortcut', from_id))
                prev_width = out_widths[from_id]
                prev_height = out_heights[from_id]
                prev_filters = out_filters[from_id]
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'connected':
                filters = int(block['output'])
                print('%5d %-6s                            %d  ->  %3d' % (ind, 'connected', prev_filters,  filters))
                prev_filters = filters
                out_widths.append(1)
                out_heights.append(1)
                out_filters.append(prev_filters)
            else:
                print('unknown type %s' % (block['type']))

    @staticmethod
    def load_conv(buf, start, conv_model):
        num_w = conv_model.weight.numel()
        num_b = conv_model.bias.numel()
        #print("start: {}, num_w: {}, num_b: {}".format(start, num_w, num_b))
        # by ysyun, use .view_as()
        conv_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]).view_as(conv_model.bias.data));   start = start + num_b
        conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]).view_as(conv_model.weight.data)); start = start + num_w
        return start
    @staticmethod
    def convert2cpu(gpu_matrix):
        return torch.cuda.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix) if torch.cuda.is_available() else torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)
    @staticmethod
    def convert2cpu_long(gpu_matrix):
        return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

    @staticmethod
    def save_conv(fp, conv_model):
        if conv_model.bias.is_cuda:
            LoadUtils.convert2cpu(conv_model.bias.data).numpy().tofile(fp)
            LoadUtils.convert2cpu(conv_model.weight.data).numpy().tofile(fp)
        else:
            conv_model.bias.data.numpy().tofile(fp)
            conv_model.weight.data.numpy().tofile(fp)
    @staticmethod
    def load_conv_bn(buf, start, conv_model, bn_model):
        num_w = conv_model.weight.numel()
        num_b = bn_model.bias.numel()
        bn_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));     start = start + num_b
        bn_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
        bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b]));  start = start + num_b
        bn_model.running_var.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
        #conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w
        conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).view_as(conv_model.weight.data)); start = start + num_w
        return start
    @staticmethod
    def save_conv_bn(fp, conv_model, bn_model):
        if bn_model.bias.is_cuda:
            LoadUtils.convert2cpu(bn_model.bias.data).numpy().tofile(fp)
            LoadUtils.convert2cpu(bn_model.weight.data).numpy().tofile(fp)
            LoadUtils.convert2cpu(bn_model.running_mean).numpy().tofile(fp)
            LoadUtils.convert2cpu(bn_model.running_var).numpy().tofile(fp)
            LoadUtils.convert2cpu(conv_model.weight.data).numpy().tofile(fp)
        else:
            bn_model.bias.data.numpy().tofile(fp)
            bn_model.weight.data.numpy().tofile(fp)
            bn_model.running_mean.numpy().tofile(fp)
            bn_model.running_var.numpy().tofile(fp)
            conv_model.weight.data.numpy().tofile(fp)
    @staticmethod
    def load_fc(buf, start, fc_model):
        num_w = fc_model.weight.numel()
        num_b = fc_model.bias.numel()
        fc_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));     start = start + num_b
        fc_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]));   start = start + num_w
        return start
    @staticmethod
    def save_fc(fp, fc_model):
        fc_model.bias.data.numpy().tofile(fp)
        fc_model.weight.data.numpy().tofile(fp)

class BoxUtils(object):
    @staticmethod
    def bbox_iou(box1, box2, x1y1x2y2=True):
        if x1y1x2y2:
            x1_min = min(box1[0], box2[0])
            x2_max = max(box1[2], box2[2])
            y1_min = min(box1[1], box2[1])
            y2_max = max(box1[3], box2[3])
            w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
            w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        else:
            w1, h1 = box1[2], box1[3]
            w2, h2 = box2[2], box2[3]
            x1_min = min(box1[0]-w1/2.0, box2[0]-w2/2.0)
            x2_max = max(box1[0]+w1/2.0, box2[0]+w2/2.0)
            y1_min = min(box1[1]-h1/2.0, box2[1]-h2/2.0)
            y2_max = max(box1[1]+h1/2.0, box2[1]+h2/2.0)

        w_union = x2_max - x1_min
        h_union = y2_max - y1_min
        w_cross = w1 + w2 - w_union
        h_cross = h1 + h2 - h_union
        carea = 0
        if w_cross <= 0 or h_cross <= 0:
            return 0.0

        area1 = w1 * h1
        area2 = w2 * h2
        carea = w_cross * h_cross
        uarea = area1 + area2 - carea
        return float(carea/uarea)

    @staticmethod
    def multi_bbox_ious(boxes1, boxes2, x1y1x2y2=True):
        if x1y1x2y2:
            x1_min = torch.min(boxes1[0], boxes2[0])
            x2_max = torch.max(boxes1[2], boxes2[2])
            y1_min = torch.min(boxes1[1], boxes2[1])
            y2_max = torch.max(boxes1[3], boxes2[3])
            w1, h1 = boxes1[2] - boxes1[0], boxes1[3] - boxes1[1]
            w2, h2 = boxes2[2] - boxes2[0], boxes2[3] - boxes2[1]
        else:
            w1, h1 = boxes1[2], boxes1[3]
            w2, h2 = boxes2[2], boxes2[3]
            x1_min = torch.min(boxes1[0]-w1/2.0, boxes2[0]-w2/2.0)
            x2_max = torch.max(boxes1[0]+w1/2.0, boxes2[0]+w2/2.0)
            y1_min = torch.min(boxes1[1]-h1/2.0, boxes2[1]-h2/2.0)
            y2_max = torch.max(boxes1[1]+h1/2.0, boxes2[1]+h2/2.0)

        w_union = x2_max - x1_min
        h_union = y2_max - y1_min
        w_cross = w1 + w2 - w_union
        h_cross = h1 + h2 - h_union
        mask = (((w_cross <= 0) + (h_cross <= 0)) > 0)
        area1 = w1 * h1
        area2 = w2 * h2
        carea = w_cross * h_cross
        carea[mask] = 0
        uarea = area1 + area2 - carea
        return carea/uarea

    @staticmethod
    def nms(boxes, nms_thresh):
        if len(boxes) == 0:
            return boxes

        det_confs = torch.zeros(len(boxes))
        for i in range(len(boxes)):
            det_confs[i] = 1-boxes[i][4]

        _,sortIds = torch.sort(det_confs)
        out_boxes = []
        for i in range(len(boxes)):
            box_i = boxes[sortIds[i]]
            if box_i[4] > 0:
                out_boxes.append(box_i)
                for j in range(i+1, len(boxes)):
                    box_j = boxes[sortIds[j]]
                    if BoxUtils.bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                        #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                        box_j[4] = 0
        return out_boxes

    @staticmethod
    def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1, validation=False, use_cuda=True):
        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        anchors = anchors.to(device)
        anchor_step = anchors.size(0)//num_anchors
        if output.dim() == 3:
            output = output.unsqueeze(0)
        batch = output.size(0)
        assert(output.size(1) == (5+num_classes)*num_anchors)
        h = output.size(2)
        w = output.size(3)
        cls_anchor_dim = batch*num_anchors*h*w

        all_boxes = []
        output = output.view(batch*num_anchors, 5+num_classes, h*w).transpose(0,1).contiguous().view(5+num_classes, cls_anchor_dim)

        grid_x = torch.linspace(0, w-1, w).repeat(batch*num_anchors, h, 1).view(cls_anchor_dim).to(device)
        grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(cls_anchor_dim).to(device)
        ix = torch.LongTensor(range(0,2)).to(device)
        anchor_w = anchors.view(num_anchors, anchor_step).index_select(1, ix[0]).repeat(1, batch, h*w).view(cls_anchor_dim)
        anchor_h = anchors.view(num_anchors, anchor_step).index_select(1, ix[1]).repeat(1, batch, h*w).view(cls_anchor_dim)

        xs, ys = torch.sigmoid(output[0]) + grid_x, torch.sigmoid(output[1]) + grid_y
        ws, hs = torch.exp(output[2]) * anchor_w.detach(), torch.exp(output[3]) * anchor_h.detach()
        det_confs = torch.sigmoid(output[4])

        # by ysyun, dim=1 means input is 2D or even dimension else dim=0
        cls_confs = torch.nn.Softmax(dim=1)(output[5:5+num_classes].transpose(0,1)).detach()
        cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
        cls_max_confs = cls_max_confs.view(-1)
        cls_max_ids = cls_max_ids.view(-1)

        sz_hw = h*w
        sz_hwa = sz_hw*num_anchors
        det_confs = LoadUtils.convert2cpu(det_confs)
        cls_max_confs = LoadUtils.convert2cpu(cls_max_confs)
        cls_max_ids = LoadUtils.convert2cpu_long(cls_max_ids)
        xs, ys = LoadUtils.convert2cpu(xs), LoadUtils.convert2cpu(ys)
        ws, hs = LoadUtils.convert2cpu(ws), LoadUtils.convert2cpu(hs)
        if validation:
            cls_confs = LoadUtils.convert2cpu(cls_confs.view(-1, num_classes))

        for b in range(batch):
            boxes = []
            for cy in range(h):
                for cx in range(w):
                    for i in range(num_anchors):
                        ind = b*sz_hwa + i*sz_hw + cy*w + cx
                        det_conf =  det_confs[ind]
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
                            box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
                            if (not only_objectness) and validation:
                                for c in range(num_classes):
                                    tmp_conf = cls_confs[ind][c]
                                    if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                        box.append(tmp_conf)
                                        box.append(c)
                            boxes.append(box)
            all_boxes.append(boxes)
        return all_boxes

class ImageUtils(object):
    @staticmethod
    def read_truths_args(lab_path, min_box_scale):
        truths = ImageUtils.read_truths(lab_path)
        new_truths = []
        for i in range(truths.shape[0]):
            if truths[i][3] < min_box_scale:
                continue
            new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4]])
        return np.array(new_truths)

    @staticmethod
    def read_truths(lab_path):
        if not os.path.exists(lab_path):
            return np.array([])
        if os.path.getsize(lab_path):
            truths = np.loadtxt(lab_path)
            truths = truths.reshape(truths.size//5, 5) # to avoid single truth problem
            return truths
        else:
            return np.array([])
    @staticmethod
    def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure):
        labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')

        ## data augmentation
        img = Image.open(imgpath).convert('RGB')
        img,flip,dx,dy,sx,sy = ImageUtils.data_augmentation(img, shape, jitter, hue, saturation, exposure)
        label = ImageUtils.fill_truth_detection(labpath, img.width, img.height, flip, dx, dy, 1./sx, 1./sy)
        return img,label

    @staticmethod
    def data_augmentation(img, shape, jitter, hue, saturation, exposure):
        oh = img.height
        ow = img.width

        dw =int(ow*jitter)
        dh =int(oh*jitter)

        pleft  = random.randint(-dw, dw)
        pright = random.randint(-dw, dw)
        ptop   = random.randint(-dh, dh)
        pbot   = random.randint(-dh, dh)

        swidth =  ow - pleft - pright
        sheight = oh - ptop - pbot

        sx = float(swidth)  / ow
        sy = float(sheight) / oh

        flip = random.randint(1,10000)%2
        cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

        dx = (float(pleft)/ow)/sx
        dy = (float(ptop) /oh)/sy

        sized = cropped.resize(shape)

        if flip:
            sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
        img = ImageUtils.random_distort_image(sized, hue, saturation, exposure)

        return img, flip, dx,dy,sx,sy

    @staticmethod
    def random_distort_image(im, hue, saturation, exposure):
        dhue = random.uniform(-hue, hue)
        dsat = ImageUtils.rand_scale(saturation)
        dexp = ImageUtils.rand_scale(exposure)
        res = ImageUtils.distort_image(im, dhue, dsat, dexp)
        return res

    @staticmethod
    def distort_image(im, hue, sat, val):
        im = im.convert('HSV')
        cs = list(im.split())
        cs[1] = cs[1].point(lambda i: i * sat)
        cs[2] = cs[2].point(lambda i: i * val)

        def change_hue(x):
            x += hue*255
            if x > 255:
                x -= 255
            if x < 0:
                x += 255
            return x
        cs[0] = cs[0].point(change_hue)
        im = Image.merge(im.mode, tuple(cs))

        im = im.convert('RGB')
        #constrain_image(im)
        return im
    @staticmethod
    def rand_scale(s):
        scale = random.uniform(1, s)
        if(random.randint(1,10000)%2):
            return scale
        return 1./scale

    @staticmethod
    def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
        max_boxes = 50
        label = np.zeros((max_boxes,5))
        if os.path.getsize(labpath):
            bs = np.loadtxt(labpath)
            if bs is None:
                return label
            bs = np.reshape(bs, (-1, 5))
            cc = 0
            for i in range(bs.shape[0]):
                x1 = bs[i][1] - bs[i][3]/2
                y1 = bs[i][2] - bs[i][4]/2
                x2 = bs[i][1] + bs[i][3]/2
                y2 = bs[i][2] + bs[i][4]/2

                x1 = min(0.999, max(0, x1 * sx - dx))
                y1 = min(0.999, max(0, y1 * sy - dy))
                x2 = min(0.999, max(0, x2 * sx - dx))
                y2 = min(0.999, max(0, y2 * sy - dy))

                bs[i][1] = (x1 + x2)/2
                bs[i][2] = (y1 + y2)/2
                bs[i][3] = (x2 - x1)
                bs[i][4] = (y2 - y1)

                if flip:
                    bs[i][1] =  0.999 - bs[i][1]

                if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                    continue
                label[cc] = bs[i]
                cc += 1
                if cc >= 50:
                    break

        label = np.reshape(label, (-1))
        return label