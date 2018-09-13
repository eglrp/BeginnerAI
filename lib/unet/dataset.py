import torch
import os
import numpy as np
from scipy.misc import imread, imresize

class VOCSegDataSet(torch.utils.data.Dataset):
    def __init__(self, root='/input/VOC2012', phase="train",
                 is_transform=False,
                 img_size=(512, 512), augmentations=None, img_norm=True):
        self.phase = phase
        self.is_transform = is_transform
        self.img_size = img_size
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.root = root
        self.infoFile = [os.path.join(self.root, "ImageSets", "Segmentation", "voc_%s.txt" % phase),
                         os.path.join(self.root, "ImageSets", "Segmentation", "benchmark_%s.txt" % phase)]

        lines = tuple(open(self.infoFile[0], 'r'))
        lines1 = tuple(open(self.infoFile[1], 'r'))
        self.file_list = [id.rstrip() for id in lines] + [id.rstrip() for id in lines1]
        self.mean = np.array([104.00699, 116.66877, 122.67892])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_index = self.file_list[index]
        im_path = os.path.join(self.root,  'JPEGImages',  file_index + '.jpg')
        lbl_path = os.path.join(self.root, 'SegmentationImages', file_index + '.png')
        im = imread(im_path)
        im = np.array(im, dtype=np.uint8)
        lbl = imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int8)
        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl

    def transform(self, img, lbl):
        img = imresize(img, (self.img_size[0], self.img_size[1])) # uint8 with RGB mode
        img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        lbl[lbl==255] = 0
        lbl = lbl.astype(float)
        lbl = imresize(lbl, (self.img_size[0], self.img_size[1]), 'nearest',
                       mode='F')
        lbl = lbl.astype(int)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def decode_segmap(self, label_mask):
        label_colours = self.get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb