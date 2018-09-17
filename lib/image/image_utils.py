import numpy as np
import random
import cv2
import os
import torch as t

class ImageUtils(object):
    @staticmethod
    def flip(image): # 左右翻转
        im_lr = np.fliplr(image).copy()
        return im_lr

    @staticmethod
    def Scale(image): #固定高度，伸缩宽度，伸缩比例0.6-1.4
        scale = random.uniform(0.6,1.4)
        height,width,c = image.shape
        image = cv2.resize(image,(int(width*scale),height))
        return image

    @staticmethod
    def Blur(image): # 模糊图像
        image = cv2.blur(image,(5,5))
        return image

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

image = cv2.imread("09.jpg")
image = ImageUtils.flip(image)
image = ImageUtils.Scale(image)
image = ImageUtils.Shift(image)
cv2.imwrite('test.jpg', image)
