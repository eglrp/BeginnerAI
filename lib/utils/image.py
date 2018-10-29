'''
@author: JJZHK
@license: (C) Copyright 2017-2023, Node Supply Chain Manager Corporation Limited.
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: aaa.py
@time: 2018/10/25 14:11
@desc: 图片操作常用方法
'''

import numpy as np
import random
import cv2
import PIL.Image as pil_image

class ConvertColor:
    def __init__(self, current='BGR', to='HSV', keep_3ch=True):
        if not ((current in {'BGR', 'RGB', 'HSV'}) and (to in {'RGB', 'HSV', 'GRAY'})):
            raise NotImplementedError
        self.current = current
        self.to = to
        self.keep_3ch = keep_3ch

    def __call__(self, image, labels=None):
        if self.current == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.to == "HSV":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif self.to == "GRAY":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                if self.keep_3ch:
                    image = np.stack([image] * 3, axis=-1)
        elif self.current == "RGB":
            if self.to == "HSV":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif self.to == "GRAY":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                if self.keep_3ch:
                    image = np.stack([image] * 3, axis=-1)
        elif self.current == "HSV":
            if self.to == "RGB":
                image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
            elif self.to == "GRAY":
                image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                if self.keep_3ch:
                    image = np.stack([image] * 3, axis=-1)
        if labels is None:
            return image
        else:
            return image, labels

class ConvertDataType:
    def __init__(self, to='uint8'):
        if not (to == 'uint8' or to == 'float32'):
            raise ValueError("`to` can be either of 'uint8' or 'float32'.")
        self.to = to

    def __call__(self, image, labels=None):
        if self.to == 'uint8':
            image = np.round(image, decimals=0).astype(np.uint8)
        else:
            image = image.astype(np.float32)
        if labels is None:
            return image
        else:
            return image, labels

class ConvertTo3Channels:
    def __init__(self):
        pass

    def __call__(self, image, labels=None):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:,:,:3]
        if labels is None:
            return image
        else:
            return image, labels

class Hue:
    '''
    Changes the hue of HSV images.

    Important:
        - Expects HSV input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, delta):
        '''
        Arguments:
            delta (int): An integer in the closed interval `[-180, 180]` that determines the hue change, where
                a change by integer `delta` means a change by `2 * delta` degrees. Read up on the HSV color format
                if you need more information.
        '''
        if not (-180 <= delta <= 180): raise ValueError("`delta` must be in the closed interval `[-180, 180]`.")
        self.delta = delta

    def __call__(self, image, labels=None):
        image[:, :, 0] = (image[:, :, 0] + self.delta) % 180.0
        if labels is None:
            return image
        else:
            return image, labels

class RandomHue:
    '''
    Randomly changes the hue of HSV images.

    Important:
        - Expects HSV input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, max_delta=18, prob=0.5):
        '''
        Arguments:
            max_delta (int): An integer in the closed interval `[0, 180]` that determines the maximal absolute
                hue change.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        if not (0 <= max_delta <= 180): raise ValueError("`max_delta` must be in the closed interval `[0, 180]`.")
        self.max_delta = max_delta
        self.prob = prob
        self.change_hue = Hue(delta=0)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.change_hue.delta = np.random.uniform(-self.max_delta, self.max_delta)
            return self.change_hue(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Saturation:
    '''
    Changes the saturation of HSV images.

    Important:
        - Expects HSV input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, factor):
        '''
        Arguments:
            factor (float): A float greater than zero that determines saturation change, where
                values less than one result in less saturation and values greater than one result
                in more saturation.
        '''
        if factor <= 0.0: raise ValueError("It must be `factor > 0`.")
        self.factor = factor

    def __call__(self, image, labels=None):
        image[:,:,1] = np.clip(image[:,:,1] * self.factor, 0, 255)
        if labels is None:
            return image
        else:
            return image, labels

class RandomSaturation:
    '''
    Randomly changes the saturation of HSV images.

    Important:
        - Expects HSV input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, lower=0.3, upper=2.0, prob=0.5):
        '''
        Arguments:
            lower (float, optional): A float greater than zero, the lower bound for the random
                saturation change.
            upper (float, optional): A float greater than zero, the upper bound for the random
                saturation change. Must be greater than `lower`.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = lower
        self.upper = upper
        self.prob = prob
        self.change_saturation = Saturation(factor=1.0)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.change_saturation.factor = np.random.uniform(self.lower, self.upper)
            return self.change_saturation(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Brightness:
    '''
    Changes the brightness of RGB images.

    Important:
        - Expects RGB input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, delta):
        '''
        Arguments:
            delta (int): An integer, the amount to add to or subtract from the intensity
                of every pixel.
        '''
        self.delta = delta

    def __call__(self, image, labels=None):
        image = np.clip(image + self.delta, 0, 255)
        if labels is None:
            return image
        else:
            return image, labels

class RandomBrightness:
    '''
    Randomly changes the brightness of RGB images.

    Important:
        - Expects RGB input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, lower=-84, upper=84, prob=0.5):
        '''
        Arguments:
            lower (int, optional): An integer, the lower bound for the random brightness change.
            upper (int, optional): An integer, the upper bound for the random brightness change.
                Must be greater than `lower`.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = float(lower)
        self.upper = float(upper)
        self.prob = prob
        self.change_brightness = Brightness(delta=0)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.change_brightness.delta = np.random.uniform(self.lower, self.upper)
            return self.change_brightness(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Contrast:
    '''
    Changes the contrast of RGB images.

    Important:
        - Expects RGB input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, factor):
        '''
        Arguments:
            factor (float): A float greater than zero that determines contrast change, where
                values less than one result in less contrast and values greater than one result
                in more contrast.
        '''
        if factor <= 0.0: raise ValueError("It must be `factor > 0`.")
        self.factor = factor

    def __call__(self, image, labels=None):
        image = np.clip(127.5 + self.factor * (image - 127.5), 0, 255)
        if labels is None:
            return image
        else:
            return image, labels

class RandomContrast:
    '''
    Randomly changes the contrast of RGB images.

    Important:
        - Expects RGB input.
        - Expects input array to be of `dtype` `float`.
    '''
    def __init__(self, lower=0.5, upper=1.5, prob=0.5):
        '''
        Arguments:
            lower (float, optional): A float greater than zero, the lower bound for the random
                contrast change.
            upper (float, optional): A float greater than zero, the upper bound for the random
                contrast change. Must be greater than `lower`.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = lower
        self.upper = upper
        self.prob = prob
        self.change_contrast = Contrast(factor=1.0)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.change_contrast.factor = np.random.uniform(self.lower, self.upper)
            return self.change_contrast(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Gamma:
    '''
    Changes the gamma value of RGB images.

    Important: Expects RGB input.
    '''
    def __init__(self, gamma):
        '''
        Arguments:
            gamma (float): A float greater than zero that determines gamma change.
        '''
        if gamma <= 0.0: raise ValueError("It must be `gamma > 0`.")
        self.gamma = gamma
        self.gamma_inv = 1.0 / gamma
        # Build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values.
        self.table = np.array([((i / 255.0) ** self.gamma_inv) * 255 for i in np.arange(0, 256)]).astype("uint8")

    def __call__(self, image, labels=None):
        image = cv2.LUT(image, self.table)
        if labels is None:
            return image
        else:
            return image, labels

class RandomGamma:
    '''
    Randomly changes the gamma value of RGB images.

    Important: Expects RGB input.
    '''
    def __init__(self, lower=0.25, upper=2.0, prob=0.5):
        '''
        Arguments:
            lower (float, optional): A float greater than zero, the lower bound for the random
                gamma change.
            upper (float, optional): A float greater than zero, the upper bound for the random
                gamma change. Must be greater than `lower`.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        if lower >= upper: raise ValueError("`upper` must be greater than `lower`.")
        self.lower = lower
        self.upper = upper
        self.prob = prob

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            gamma = np.random.uniform(self.lower, self.upper)
            change_gamma = Gamma(gamma=gamma)
            return change_gamma(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class HistogramEqualization:
    '''
    Performs histogram equalization on HSV images.

    Importat: Expects HSV input.
    '''
    def __init__(self):
        pass

    def __call__(self, image, labels=None):
        image[:,:,2] = cv2.equalizeHist(image[:,:,2])
        if labels is None:
            return image
        else:
            return image, labels

class RandomHistogramEqualization:
    '''
    Randomly performs histogram equalization on HSV images. The randomness only refers
    to whether or not the equalization is performed.

    Importat: Expects HSV input.
    '''
    def __init__(self, prob=0.5):
        '''
        Arguments:
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        self.prob = prob
        self.equalize = HistogramEqualization()

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            return self.equalize(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class ChannelSwap:
    '''
    Swaps the channels of images.
    '''
    def __init__(self, order):
        '''
        Arguments:
            order (tuple): A tuple of integers that defines the desired channel order
                of the input images after the channel swap.
        '''
        self.order = order

    def __call__(self, image, labels=None):
        image = image[:,:,self.order]
        if labels is None:
            return image
        else:
            return image, labels

class RandomChannelSwap:
    '''
    Randomly swaps the channels of RGB images.

    Important: Expects RGB input.
    '''
    def __init__(self, prob=0.5):
        '''
        Arguments:
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
        '''
        self.prob = prob
        # All possible permutations of the three image channels except the original order.
        self.permutations = ((0, 2, 1),
                             (1, 0, 2), (1, 2, 0),
                             (2, 0, 1), (2, 1, 0))
        self.swap_channels = ChannelSwap(order=(0, 1, 2))

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            i = np.random.randint(5) # There are 6 possible permutations.
            self.swap_channels.order = self.permutations[i]
            return self.swap_channels(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Flip:
    '''
    Flips images horizontally or vertically.
    '''
    def __init__(self,
                 dim='horizontal',
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            dim (str, optional): Can be either of 'horizontal' and 'vertical'.
                If 'horizontal', images will be flipped horizontally, i.e. along
                the vertical axis. If 'horizontal', images will be flipped vertically,
                i.e. along the horizontal axis.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        if not (dim in {'horizontal', 'vertical'}): raise ValueError("`dim` can be one of 'horizontal' and 'vertical'.")
        self.dim = dim
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        if self.dim == 'horizontal':
            image = image[:,::-1]
            if labels is None:
                return image
            else:
                labels = np.copy(labels)
                labels[:, [xmin, xmax]] = img_width - labels[:, [xmax, xmin]]
                return image, labels
        else:
            image = image[::-1]
            if labels is None:
                return image
            else:
                labels = np.copy(labels)
                labels[:, [ymin, ymax]] = img_height - labels[:, [ymax, ymin]]
                return image, labels

class RandomFlip:
    '''
    Randomly flips images horizontally or vertically. The randomness only refers
    to whether or not the image will be flipped.
    '''
    def __init__(self,
                 dim='horizontal',
                 prob=0.5,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            dim (str, optional): Can be either of 'horizontal' and 'vertical'.
                If 'horizontal', images will be flipped horizontally, i.e. along
                the vertical axis. If 'horizontal', images will be flipped vertically,
                i.e. along the horizontal axis.
            prob (float, optional): `(1 - prob)` determines the probability with which the original,
                unaltered image is returned.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        self.dim = dim
        self.prob = prob
        self.labels_format = labels_format
        self.flip = Flip(dim=self.dim, labels_format=self.labels_format)

    def __call__(self, image, labels=None):
        p = np.random.uniform(0,1)
        if p >= (1.0-self.prob):
            self.flip.labels_format = self.labels_format
            return self.flip(image, labels)
        elif labels is None:
            return image
        else:
            return image, labels

class Resize:
    '''
    Resizes images to a specified height and width in pixels.
    '''

    def __init__(self,
                 height,
                 width,
                 interpolation_mode=cv2.INTER_LINEAR,
                 box_filter=None,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            height (int): The desired height of the output images in pixels.
            width (int): The desired width of the output images in pixels.
            interpolation_mode (int, optional): An integer that denotes a valid
                OpenCV interpolation mode. For example, integers 0 through 5 are
                valid interpolation modes.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        # if not (isinstance(box_filter, BoxFilter) or box_filter is None):
        #     raise ValueError("`box_filter` must be either `None` or a `BoxFilter` object.")
        self.out_height = height
        self.out_width = width
        self.interpolation_mode = interpolation_mode
        self.box_filter = box_filter
        self.labels_format = labels_format

    def __call__(self, image, labels=None, return_inverter=False):

        img_height, img_width = image.shape[:2]

        xmin = self.labels_format['xmin']
        ymin = self.labels_format['ymin']
        xmax = self.labels_format['xmax']
        ymax = self.labels_format['ymax']

        image = cv2.resize(image,
                           dsize=(self.out_width, self.out_height),
                           interpolation=self.interpolation_mode)

        if return_inverter:
            def inverter(labels):
                labels = np.copy(labels)
                labels[:, [ymin+1, ymax+1]] = np.round(labels[:, [ymin+1, ymax+1]] * (img_height / self.out_height), decimals=0)
                labels[:, [xmin+1, xmax+1]] = np.round(labels[:, [xmin+1, xmax+1]] * (img_width / self.out_width), decimals=0)
                return labels

        if labels is None:
            if return_inverter:
                return image, inverter
            else:
                return image
        else:
            labels = np.copy(labels)
            labels[:, [ymin, ymax]] = np.round(labels[:, [ymin, ymax]] * (self.out_height / img_height), decimals=0)
            labels[:, [xmin, xmax]] = np.round(labels[:, [xmin, xmax]] * (self.out_width / img_width), decimals=0)

            if not (self.box_filter is None):
                self.box_filter.labels_format = self.labels_format
                labels = self.box_filter(labels=labels,
                                         image_height=self.out_height,
                                         image_width=self.out_width)

            if return_inverter:
                return image, labels, inverter
            else:
                return image, labels

class ResizeRandomInterp:
    '''
    Resizes images to a specified height and width in pixels using a radnomly
    selected interpolation mode.
    '''

    def __init__(self,
                 height,
                 width,
                 interpolation_modes=[cv2.INTER_NEAREST,
                                      cv2.INTER_LINEAR,
                                      cv2.INTER_CUBIC,
                                      cv2.INTER_AREA,
                                      cv2.INTER_LANCZOS4],
                 box_filter=None,
                 labels_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):
        '''
        Arguments:
            height (int): The desired height of the output image in pixels.
            width (int): The desired width of the output image in pixels.
            interpolation_modes (list/tuple, optional): A list/tuple of integers
                that represent valid OpenCV interpolation modes. For example,
                integers 0 through 5 are valid interpolation modes.
            box_filter (BoxFilter, optional): Only relevant if ground truth bounding boxes are given.
                A `BoxFilter` object to filter out bounding boxes that don't meet the given criteria
                after the transformation. Refer to the `BoxFilter` documentation for details. If `None`,
                the validity of the bounding boxes is not checked.
            labels_format (dict, optional): A dictionary that defines which index in the last axis of the labels
                of an image contains which bounding box coordinate. The dictionary maps at least the keywords
                'xmin', 'ymin', 'xmax', and 'ymax' to their respective indices within last axis of the labels array.
        '''
        if not (isinstance(interpolation_modes, (list, tuple))):
            raise ValueError("`interpolation_mode` must be a list or tuple.")
        self.height = height
        self.width = width
        self.interpolation_modes = interpolation_modes
        self.box_filter = box_filter
        self.labels_format = labels_format
        self.resize = Resize(height=self.height,
                             width=self.width,
                             box_filter=self.box_filter,
                             labels_format=self.labels_format)

    def __call__(self, image, labels=None, return_inverter=False):
        self.resize.interpolation_mode = np.random.choice(self.interpolation_modes)
        self.resize.labels_format = self.labels_format
        return self.resize(image, labels, return_inverter)


class ImageUtils(object):
    '''
    image - BGR 模式图片
    '''
    @staticmethod
    def flip(image): # 左右翻转
        im_lr = np.fliplr(image).copy()
        return im_lr

    '''
    image - BGR 模式图片
    '''
    @staticmethod
    def Scale(image): #固定高度，伸缩宽度，伸缩比例0.6-1.4
        scale = random.uniform(0.6,1.4)
        height,width,c = image.shape
        image = cv2.resize(image,(int(width*scale),height))
        return image

    '''
    image - BGR 模式图片
    '''
    @staticmethod
    def Blur(image): # 模糊图像
        image = cv2.blur(image,(5,5))
        return image

    '''
    image - BGR 模式图片
    '''
    @staticmethod
    def Brightness(image): # 图像变亮
        hsv = ImageUtils.BGR2HSV(image)
        h,s,v = cv2.split(hsv)
        adjust = random.choice([0.5,1.5])
        v = v*adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h,s,v))
        image = ImageUtils.HSV2BGR(hsv)
        return image
    @staticmethod
    def BGR2RGB(img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    @staticmethod
    def BGR2HSV(img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    @staticmethod
    def HSV2BGR(img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)

    '''
    image - BGR 模式图片
    '''
    @staticmethod
    def Hue(image): #色调
        hsv = ImageUtils.BGR2HSV(image)
        h,s,v = cv2.split(hsv)
        adjust = random.choice([0.5,1.5])
        h = h*adjust
        h = np.clip(h, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h,s,v))
        image = ImageUtils.HSV2BGR(hsv)
        return image

    '''
    image - BGR 模式图片
    '''
    @staticmethod
    def Saturation(image): # 饱和度
        hsv = ImageUtils.BGR2HSV(image)
        h,s,v = cv2.split(hsv)
        adjust = random.choice([0.5,1.5])
        s = s*adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h,s,v))
        image = ImageUtils.HSV2BGR(hsv)
        return image

    '''
    image - BGR 模式图片
    '''
    @staticmethod
    def Shift(image): #平移变换
        #平移变换
        height,width,c = image.shape
        after_shfit_image = np.zeros((height,width,c),dtype=image.dtype)
        after_shfit_image[:,:,:] = (104,117,123) #bgr
        shift_x = random.uniform(-width*0.2,width*0.2)
        shift_y = random.uniform(-height*0.2,height*0.2)
        #原图像的平移
        if shift_x>=0 and shift_y>=0:
            after_shfit_image[int(shift_y):,int(shift_x):,:] = image[:height-int(shift_y),:width-int(shift_x),:]
        elif shift_x>=0 and shift_y<0:
            after_shfit_image[:height+int(shift_y),int(shift_x):,:] = image[-int(shift_y):,:width-int(shift_x),:]
        elif shift_x <0 and shift_y >=0:
            after_shfit_image[int(shift_y):,:width+int(shift_x),:] = image[:height-int(shift_y),-int(shift_x):,:]
        elif shift_x<0 and shift_y<0:
            after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = image[-int(shift_y):,-int(shift_x):,:]

        return after_shfit_image

    '''
    image - BGR 模式图片,
    输入为PIL.Image对象
    '''
    @staticmethod
    def CV2ToPIL(image):
        return pil_image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

    '''
    image - PIL.Image类型对象
    '''
    @staticmethod
    def PILToCV2(image):
        return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

# image = cv2.imread("09.jpg")
# image = ImageUtils.flip(image)
# image = ImageUtils.Scale(image)
# image = ImageUtils.Shift(image)
# cv2.imwrite('test.jpg', image)
