"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

class VOCAnnotationTransform(object):
    def __init__(self, VOC_CLASS, class_to_ind=None, keep_difficult=False):
        self.VOC_CLASSES = VOC_CLASS
        self.class_to_ind = class_to_ind or dict(
            zip(self.VOC_CLASSES, range(len(self.VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        res = []
        # 2008_000003.jpg 2 46 11 500 333 18 62 190 83 243 14
        num_box = int(target[1])
        for i in range(num_box):
            bndbox = []
            x = int(target[2+5*i]) - 1
            x = x / width if i % 2 == 0 else x / height
            bndbox.append(x)
            y = int(target[3+5*i]) - 1
            y = y / width if i % 2 == 0 else y / height
            bndbox.append(y)
            x2 = int(target[4+5*i]) - 1
            x2 = x2 / width if i % 2 == 0 else x2 / height
            bndbox.append(x2)
            y2 = int(target[5+5*i]) - 1
            y2 = y2 / width if i % 2 == 0 else y2 / height
            bndbox.append(y2)

            label_idx = target[6+5*i]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]

        return res  # 形状形如[[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object
    """

    def __init__(self, root="/input/VOC", file="utils/voctrain.txt", transform=None, target_transform=None):
        self.root = root
        self.transform = transform   #SSDAugmentation(cfg['min_dim'],MEANS))  图像增强
        self.target_transform = target_transform   #VOCAnnotationTransform()  注释变换

        self.ids = list()   #图像的id全部保存在ids

        self.img_ids = open(file).read().strip().split()

    def __getitem__(self, index):
        '''
        :param index: 取第几条数据
        :return: 一张图像及对应的真值框和类别
        '''
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        '''
        取某条数据
        :param index: 取第几条数据
        :return: 一张图像、对应的真值框、高、宽
        '''
        img_id = self.ids[index]
        splited = img_id.strip().split()
        img = cv2.imread(os.path.join(self.root, "JPEGImages", splited[0]))
        height, width, channels = img.shape # 得到图像的高、宽、通道（数据集中高宽不一定）

        target = ET.parse(self._annopath % img_id).getroot()

        if self.target_transform is not None:
            target = self.target_transform(splited, width, height)

        # SSDAugmentation(cfg['min_dim'],MEANS))  图像增强
        if self.transform is not None:
            # 转化为tensor  形状为（x,5）  x:图像中的物体总数   5：bbox坐标、 类别
            target = np.array(target)
            # 图像增强
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb  转化为rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            # hstack合并  axis=1按照列合并   target：一行内容是boxes坐标+类别
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        # 一张图像、对应的真值框和类别、高、宽
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
