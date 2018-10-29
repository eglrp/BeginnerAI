import keras.backend as K
import numpy as np
from keras.models import load_model
import colorsys
import random
import os
import imghdr
from PIL import Image, ImageDraw, ImageFont
import lib.utils.drawutils as draw
import lib.keras.yolov2.utils as k_utils

CONFIG = {
    "MODEL_PATH" : "utils/keras_yolov2_coco.h5",
    "ANCHORS" : "0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828",
    "CLASSES" : ["person","bicycle","car" ,"motorbike","aeroplane","bus","train","truck",
                 "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
                 "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
                 "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
                 "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
                 "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
                 "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
                 "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote",
                 "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock",
                 "vase","scissors","teddy bear","hair drier","toothbrush"],
    "TEST_PATH" : "../testImages",
    "OUTPUT_PATH" : "outputs"
}

sess = K.get_session()
class_names = CONFIG["CLASSES"]
anchors = [float(x) for x in CONFIG["ANCHORS"].split(',')]
anchors = np.array(anchors).reshape(-1, 2)
yolo_model = load_model(CONFIG["MODEL_PATH"])
num_classes = len(class_names)
num_anchors = len(anchors)

model_output_channels = yolo_model.layers[-1].output_shape[-1]
model_image_size = yolo_model.layers[0].input_shape[1:3]
is_fixed_size = model_image_size != (None, None)

# Generate colors for drawing bounding boxes.
hsv_tuples = [(x / float(len(class_names)), 1., 1.)
              for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))
random.seed(10101)  # Fixed seed for consistent colors across runs.
random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
random.seed(None)  # Reset seed to default.
yolo_outputs = k_utils.convert_result(yolo_model.output, anchors, len(class_names))
input_image_shape = K.placeholder(shape=(2, ))
boxes, scores, classes = k_utils.draw_helper(
    yolo_outputs,
    input_image_shape,
    to_threshold=0.4,
    iou_threshold=0.5)

for image_file in os.listdir(CONFIG["TEST_PATH"]):
    image_type = imghdr.what(os.path.join(CONFIG["TEST_PATH"], image_file))
    image = Image.open(os.path.join(CONFIG["TEST_PATH"], image_file))
    if is_fixed_size:  # TODO: When resizing we can use minibatch input.
        resized_image = image.resize(
            tuple(reversed(model_image_size)), Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
    else:
        # Due to skip connection + max pooling in YOLO_v2, inputs must have
        # width and height as multiples of 32.
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        resized_image = image.resize(new_image_size, Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
        print(image_data.shape)

    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes],
        feed_dict={
            yolo_model.input: image_data,
            input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })

    result = []
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        result.append(
            [
                (left, top),
                (right, bottom),
                predicted_class,
                score
            ]
        )
    draw.draw_box(image, result, "outputs/%s" % image_file, datatype="coco")
    #     label = '{} {:.2f}'.format(predicted_class, score)
    #
    #     draw = ImageDraw.Draw(image)
    #     label_size = draw.textsize(label, font)
    #
    #     top, left, bottom, right = box
    #     top = max(0, np.floor(top + 0.5).astype('int32'))
    #     left = max(0, np.floor(left + 0.5).astype('int32'))
    #     bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    #     right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    #     #the result's origin is in top left
    #     print(label, (left, top), (right, bottom), thickness)
    #
    #     if top - label_size[1] >= 0:
    #         text_origin = np.array([left, top - label_size[1]])
    #     else:
    #         text_origin = np.array([left, top + 1])
    #
    #     # My kingdom for a good redistributable image drawing library.
    #     # for i in range(thickness):
    #     #     draw.rectangle(
    #     #         [left + i, top + i, right - i, bottom - i],
    #     #         outline=colors[c])
    #     draw.rectangle(
    #         [tuple(text_origin), tuple(text_origin + label_size)],
    #         fill=colors[c])
    #     draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    #     del draw
    #
    # image.save(os.path.join(CONFIG["OUTPUT_PATH"], image_file), quality=90)
sess.close()