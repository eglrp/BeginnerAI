# coding=utf-8
import random
import os
import numpy as np
import cv2
import torch
import xml.etree.ElementTree as ET

from os.path import join
from os import listdir
from PIL import Image
from skimage.transform import resize


from torch.utils.data import Dataset
from torchvision.transforms.transforms import ToTensor, Normalize, Compose

from lib.utils.BasicDataSet import Cifar10DataSet,MnistDataSet,STLDataSet
from lib.pytorch.utils.functions import is_image_file, load_img

class DataSetFromFolderForPix2Pix(Dataset):
    def __init__(self, image_dir):
        super(DataSetFromFolderForPix2Pix, self).__init__()
        self.photo_path = join(image_dir, "A")
        self.sketch_path = join(image_dir, "B")
        self.image_filenames = [x for x in listdir(self.photo_path) if is_image_file(x) ]

        transform_list = [ToTensor(),
                          Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
        self.transform = Compose(transform_list)

    def __getitem__(self, index):
        input = load_img(join(self.photo_path, self.image_filenames[index]))
        input = self.transform(input)
        target = load_img(join(self.sketch_path, self.image_filenames[index]))
        target = self.transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)

class DataSetFromFolderForCycleGAN(Dataset):
    def __init__(self, image_dir, subfolder='train', transform=None, resize_scale=None, crop_size=None, fliplr=False):
        super(DataSetFromFolderForCycleGAN, self).__init__()
        self.input_path = join(image_dir, subfolder)
        self.image_filenames = [x for x in sorted(listdir(self.input_path))]
        self.transform = transform
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.fliplr = fliplr

    def __getitem__(self, index):
        # Load Image
        img_fn = join(self.input_path, self.image_filenames[index])
        img = load_img(img_fn)

        if self.crop_size:
            x = random.randint(0, self.resize_scale - self.crop_size + 1)
            y = random.randint(0, self.resize_scale - self.crop_size + 1)
            img = img.crop((x, y, x + self.crop_size, y + self.crop_size))
        if self.fliplr:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_filenames)

class Cifar10DataSetForPytorch(Dataset):
    def __init__(self, root="/input/cifar10/", train=True,transform=None, target_transform=None, target_label=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        reader = Cifar10DataSet(self.root, special_label=target_label)
        (self.train_data, self.train_label), (self.test_data, self.test_label) = reader.read(channel_first=False)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_label[index]
        else:
            img, target = self.test_data[index], self.test_label[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

class MNISTDataSetForPytorch(Dataset):
    def __init__(self, root="/input/mnist.npz", train=True, radio=0.9, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        reader = MnistDataSet(root=self.root, radio=radio)

        (self.train_data, self.train_labels),(self.test_data, self.test_labels) = reader.read()

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)



