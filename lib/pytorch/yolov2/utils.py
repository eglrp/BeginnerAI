import torch
import math
import random
import os
from PIL import Image
import numpy as np
class ConfigUtils(object):
    @staticmethod
    def read_data_cfg(datacfg):
        options = dict()
        with open(datacfg, 'r') as fp:
            lines = fp.readlines()

        for line in lines:
            line = line.strip()
            if line == '':
                continue
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()
            options[key] = value
        return options

    @staticmethod
    def file_lines(thefilepath):
        count = 0
        thefile = open(thefilepath, 'rb')
        while True:
            buffer = thefile.read(8192 * 1024).decode('utf-8')
            if not buffer:
                break
            count += buffer.count('\n')
        thefile.close()
        return count

    @staticmethod
    def parse_cfg(cfgfile):
        blocks = []
        fp = open(cfgfile, 'r')
        block = None
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
                key, value = line.split('=')
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
    def print_cfg(blocks):
        print('layer     filters    size              input                output')
        prev_width = 416
        prev_height = 416
        prev_filters = 3
        out_filters = []
        out_widths = []
        out_heights = []
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
                pad = (kernel_size - 1) / 2 if is_pad else 0
                width = (prev_width + 2 * pad - kernel_size) / stride + 1
                height = (prev_height + 2 * pad - kernel_size) / stride + 1
                print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'conv', filters,
                                                                                             kernel_size, kernel_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
                prev_width = width
                prev_height = height
                prev_filters = filters
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                width = prev_width / stride
                height = prev_height / stride
                print('%5d %-6s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'max',
                                                                                              pool_size, pool_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
                prev_width = width
                prev_height = height
                prev_filters = filters
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'avgpool':
                width = 1
                height = 1
                print('%5d %-6s                   %3d x %3d x%4d   ->  %3d' %
                      (ind, 'avg', prev_width, prev_height, prev_filters,  prev_filters))
                prev_width = width
                prev_height = height
                prev_filters = filters
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'softmax':
                print('%5d %-6s                                    ->  %3d' %
                      (ind, 'softmax', prev_filters))
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'cost':
                print('%5d %-6s                                     ->  %3d' %
                      (ind, 'cost', prev_filters))
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                filters = stride * stride * prev_filters
                width = prev_width / stride
                height = prev_height / stride
                print('%5d %-6s             / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' %
                      (ind, 'reorg', stride, prev_width, prev_height, prev_filters, width, height, filters))
                prev_width = width
                prev_height = height
                prev_filters = filters
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
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
            elif block['type'] == 'region':
                print('%5d %-6s' % (ind, 'detection'))
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'shortcut':
                from_id = int(block['from'])
                from_id = from_id if from_id > 0 else from_id + ind
                print('%5d %-6s %d' % (ind, 'shortcut', from_id))
                prev_width = out_widths[from_id]
                prev_height = out_heights[from_id]
                prev_filters = out_filters[from_id]
                out_widths.append(prev_width)
                out_heights.append(prev_height)
                out_filters.append(prev_filters)
            elif block['type'] == 'connected':
                filters = int(block['output'])
                print('%5d %-6s                            %d  ->  %3d' %
                      (ind, 'connected', prev_filters,  filters))
                prev_filters = filters
                out_widths.append(1)
                out_heights.append(1)
                out_filters.append(prev_filters)
            else:
                print('unknown type %s' % (block['type']))

class LoadUtils(object):
    @staticmethod
    def load_conv(buf, start, conv_model):
        num_w = conv_model.weight.numel()
        num_b = conv_model.bias.numel()
        conv_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
        start = start + num_b
        conv_model.weight.data.copy_(torch.from_numpy(
            buf[start:start + num_w]).view(conv_model.weight.data.shape))
        start = start + num_w
        return start

    @staticmethod
    def save_conv(fp, conv_model):
        if conv_model.bias.is_cuda:
            LossUtils.convert2cpu(conv_model.bias.data).numpy().tofile(fp)
            LossUtils.convert2cpu(conv_model.weight.data).numpy().tofile(fp)
        else:
            conv_model.bias.data.numpy().tofile(fp)
            conv_model.weight.data.numpy().tofile(fp)

    @staticmethod
    def load_conv_bn(buf, start, conv_model, bn_model):
        num_w = conv_model.weight.numel()
        num_b = bn_model.bias.numel()
        bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
        start = start + num_b
        bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]))
        start = start + num_b
        bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]))
        start = start + num_b
        bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]))
        start = start + num_b
        conv_model.weight.data.copy_(torch.from_numpy(
            buf[start:start + num_w]).view(conv_model.weight.data.shape))
        start = start + num_w
        return start

    @staticmethod
    def save_conv_bn(fp, conv_model, bn_model):
        if bn_model.bias.is_cuda:
            LossUtils.convert2cpu(bn_model.bias.data).numpy().tofile(fp)
            LossUtils.convert2cpu(bn_model.weight.data).numpy().tofile(fp)
            LossUtils.convert2cpu(bn_model.running_mean).numpy().tofile(fp)
            LossUtils.convert2cpu(bn_model.running_var).numpy().tofile(fp)
            LossUtils.convert2cpu(conv_model.weight.data).numpy().tofile(fp)
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
        fc_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
        start = start + num_b
        fc_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]))
        start = start + num_w
        return start

    @staticmethod
    def save_fc(fp, fc_model):
        fc_model.bias.data.numpy().tofile(fp)
        fc_model.weight.data.numpy().tofile(fp)

class LossUtils(object):
    @staticmethod
    def convert2cpu(gpu_matrix):
        return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

    @staticmethod
    def convert2cpu_long(gpu_matrix):
        return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

    @staticmethod
    def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen):
        nB = target.size(0)
        nA = num_anchors
        nC = num_classes
        anchor_step = int(len(anchors) / num_anchors)
        conf_mask = torch.ones(nB, nA, nH, nW) * noobject_scale
        coord_mask = torch.zeros(nB, nA, nH, nW)
        cls_mask = torch.zeros(nB, nA, nH, nW)
        tx = torch.zeros(nB, nA, nH, nW)
        ty = torch.zeros(nB, nA, nH, nW)
        tw = torch.zeros(nB, nA, nH, nW)
        th = torch.zeros(nB, nA, nH, nW)
        tconf = torch.zeros(nB, nA, nH, nW)
        tcls = torch.zeros(nB, nA, nH, nW)

        nAnchors = nA * nH * nW
        nPixels = nH * nW
        for b in range(nB):
            cur_pred_boxes = pred_boxes[b * nAnchors:(b + 1) * nAnchors].t()
            cur_ious = torch.zeros(nAnchors)
            for t in range(50):
                if target[b][t * 5 + 1] == 0:
                    break
                gx = target[b][t * 5 + 1] * nW
                gy = target[b][t * 5 + 2] * nH
                gw = target[b][t * 5 + 3] * nW
                gh = target[b][t * 5 + 4] * nH
                cur_gt_boxes = torch.FloatTensor(
                    [gx, gy, gw, gh]).repeat(nAnchors, 1).t()
                cur_ious = torch.max(cur_ious, BoxUtils.bbox_ious(
                    cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
            temp_thresh = cur_ious > sil_thresh
            conf_mask[b][temp_thresh.view(conf_mask[b].shape)] = 0
        if seen < 12800:
            if anchor_step == 4:
                tx = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(
                    1, torch.LongTensor([2])).view(1, nA, 1, 1).repeat(nB, 1, nH, nW)
                ty = torch.FloatTensor(anchors).view(num_anchors, anchor_step).index_select(
                    1, torch.LongTensor([2])).view(1, nA, 1, 1).repeat(nB, 1, nH, nW)
            else:
                tx.fill_(0.5)
                ty.fill_(0.5)
            tw.zero_()
            th.zero_()
            coord_mask.fill_(1)

        nGT = 0
        nCorrect = 0
        for b in range(nB):
            for t in range(50):
                if target[b][t * 5 + 1] == 0:
                    break
                nGT = nGT + 1
                best_iou = 0.0
                best_n = -1
                min_dist = 10000
                gx = target[b][t * 5 + 1] * nW
                gy = target[b][t * 5 + 2] * nH
                gi = int(gx)
                gj = int(gy)
                gw = target[b][t * 5 + 3] * nW
                gh = target[b][t * 5 + 4] * nH
                gt_box = [0, 0, gw, gh]
                for n in range(nA):
                    aw = anchors[anchor_step * n]
                    ah = anchors[anchor_step * n + 1]
                    anchor_box = [0, 0, aw, ah]
                    iou = BoxUtils.bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                    if anchor_step == 4:
                        ax = anchors[anchor_step * n + 2]
                        ay = anchors[anchor_step * n + 3]
                        dist = pow(((gi + ax) - gx), 2) + pow(((gj + ay) - gy), 2)
                    if iou > best_iou:
                        best_iou = iou
                        best_n = n
                    elif anchor_step == 4 and iou == best_iou and dist < min_dist:
                        best_iou = iou
                        best_n = n
                        min_dist = dist

                gt_box = [gx, gy, gw, gh]
                pred_box = pred_boxes[b * nAnchors +
                                      best_n * nPixels + gj * nW + gi]

                coord_mask[b][best_n][gj][gi] = 1
                cls_mask[b][best_n][gj][gi] = 1
                conf_mask[b][best_n][gj][gi] = object_scale
                tx[b][best_n][gj][gi] = target[b][t * 5 + 1] * nW - gi
                ty[b][best_n][gj][gi] = target[b][t * 5 + 2] * nH - gj
                tw[b][best_n][gj][gi] = math.log(
                    gw / anchors[anchor_step * best_n])
                th[b][best_n][gj][gi] = math.log(
                    gh / anchors[anchor_step * best_n + 1])

                gt_box = torch.Tensor(gt_box)
                iou = BoxUtils.bbox_iou(gt_box, pred_box,
                               x1y1x2y2=False)  # best_iou
                tconf[b][best_n][gj][gi] = iou
                tcls[b][best_n][gj][gi] = target[b][t * 5]
                if iou > 0.5:
                    nCorrect = nCorrect + 1

        return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls

class BoxUtils(object):
    @staticmethod
    def bbox_iou(box1, box2, x1y1x2y2=True):
        if x1y1x2y2:
            mx = min(box1[0], box2[0])
            Mx = max(box1[2], box2[2])
            my = min(box1[1], box2[1])
            My = max(box1[3], box2[3])
            w1 = box1[2] - box1[0]
            h1 = box1[3] - box1[1]
            w2 = box2[2] - box2[0]
            h2 = box2[3] - box2[1]
        else:
            mx = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
            Mx = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
            my = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
            My = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
            w1 = box1[2]
            h1 = box1[3]
            w2 = box2[2]
            h2 = box2[3]
        uw = Mx - mx
        uh = My - my
        cw = w1 + w2 - uw
        ch = h1 + h2 - uh
        carea = 0
        if cw <= 0 or ch <= 0:
            return 0.0

        area1 = w1 * h1
        area2 = w2 * h2
        carea = cw * ch
        uarea = area1 + area2 - carea
        return carea / uarea

    @staticmethod
    def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
        if x1y1x2y2:
            mx = torch.min(boxes1[0], boxes2[0])
            Mx = torch.max(boxes1[2], boxes2[2])
            my = torch.min(boxes1[1], boxes2[1])
            My = torch.max(boxes1[3], boxes2[3])
            w1 = boxes1[2] - boxes1[0]
            h1 = boxes1[3] - boxes1[1]
            w2 = boxes2[2] - boxes2[0]
            h2 = boxes2[3] - boxes2[1]
        else:
            mx = torch.min(boxes1[0] - boxes1[2] / 2.0,
                           boxes2[0] - boxes2[2] / 2.0)
            Mx = torch.max(boxes1[0] + boxes1[2] / 2.0,
                           boxes2[0] + boxes2[2] / 2.0)
            my = torch.min(boxes1[1] - boxes1[3] / 2.0,
                           boxes2[1] - boxes2[3] / 2.0)
            My = torch.max(boxes1[1] + boxes1[3] / 2.0,
                           boxes2[1] + boxes2[3] / 2.0)
            w1 = boxes1[2]
            h1 = boxes1[3]
            w2 = boxes2[2]
            h2 = boxes2[3]
        uw = Mx - mx
        uh = My - my
        cw = w1 + w2 - uw
        ch = h1 + h2 - uh
        mask = ((cw <= 0) + (ch <= 0) > 0)
        area1 = w1 * h1
        area2 = w2 * h2
        carea = cw * ch
        carea[mask] = 0
        uarea = area1 + area2 - carea
        return carea / uarea

class ImageUtils(object):
    @staticmethod
    def scale_image_channel(im, c, v):
        cs = list(im.split())
        cs[c] = cs[c].point(lambda i: i * v)
        out = Image.merge(im.mode, tuple(cs))
        return out

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
    def random_distort_image(im, hue, saturation, exposure):
        dhue = random.uniform(-hue, hue)
        dsat = ImageUtils.rand_scale(saturation)
        dexp = ImageUtils.rand_scale(exposure)
        res = ImageUtils.distort_image(im, dhue, dsat, dexp)
        return res

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

    @staticmethod
    def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure):
        labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')

        ## data augmentation
        img = Image.open(imgpath).convert('RGB')
        img,flip,dx,dy,sx,sy = ImageUtils.data_augmentation(img, shape, jitter, hue, saturation, exposure)
        label = ImageUtils.fill_truth_detection(labpath, img.width, img.height, flip, dx, dy, 1./sx, 1./sy)
        return img,label

    @staticmethod
    def read_truths_args(lab_path, min_box_scale):
        truths = ImageUtils.read_truths(lab_path)
        new_truths = []
        for i in range(truths.shape[0]):
            if truths[i][3] < min_box_scale:
                continue
            new_truths.append([truths[i][0], truths[i][1],
                               truths[i][2], truths[i][3], truths[i][4]])
        return np.array(new_truths)

    @staticmethod
    def read_truths(lab_path):
        if not os.path.exists(lab_path):
            return np.array([])
        if os.path.getsize(lab_path):
            truths = np.loadtxt(lab_path)
            # to avoid single truth problem
            truths = truths.reshape(truths.size / 5, 5)
            return truths
        else:
            return np.array([])