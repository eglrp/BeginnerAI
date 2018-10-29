'''
@author: JJZHK
@license: (C) Copyright 2017-2023, Node Supply Chain Manager Corporation Limited.
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: aaa.py
@time: 2018/10/25 14:11
@desc:
'''
import cv2
import numpy as np
import PIL.ImageDraw as pil_imagedraw
import PIL.ImageFont as pil_imagefont
import PIL.Image as pil_image
import lib.utils.image as lui
import lib.utils.Config as luc

'''
image : PIL.Image类型对象
'''
def draw_box(image, boxes, outputs, datatype="voc"):
    DATA_CONFIG = luc.TOTAL_CONFIG[datatype]
    COLOR = DATA_CONFIG["BOX_COLORS"]
    CLASS = DATA_CONFIG["CLASSES"]

    font = pil_imagefont.truetype(font='Courier.dfont',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    for (x1, y1), (x2,y2), class_name, prob in boxes:
        draw = pil_imagedraw.Draw(image)
        left = int(x1)
        top = int(y1)
        right = int(x2)
        bottom = int(y2)
        color = tuple(COLOR[CLASS.index(class_name)])

        label = '{} {:.2f}'.format(class_name, prob)
        label_size = draw.textsize(label, font)

        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i], outline=color)

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=color)
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    image.save(outputs)

    return image

def draw_box_by_cv2(image, boxes, outputs, datatype="voc"):
    img = lui.ImageUtils.CV2ToPIL(image)
    img = draw_box(img, boxes, outputs, datatype)
    return lui.ImageUtils.PILToCV2(img)