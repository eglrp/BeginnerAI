import keras
import tensorflow as tf
import numpy as np
import os
import colorsys
import PIL
import PIL.ImageFont
import PIL.ImageDraw
import lib.utils.drawutils as lud
class YOLO(object):
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG
        self.model_path = self.CONFIG["MODEL_PATH"]
        self.class_names = CONFIG["CLASSES"]
        self.score = CONFIG["SCORE"]
        self.iou = CONFIG["IOU"]
        self.model_image_size = (CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])
        self.anchors = self._get_anchors()
        self.sess = keras.backend.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_anchors(self):
        anchors = [float(x) for x in self.CONFIG["ANCHORS"].split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'keras.backend.ras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        self.yolo_model = keras.models.load_model(model_path)
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = keras.backend.placeholder(shape=(2, ))

        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, saveto):

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                keras.backend.learning_phase(): 0
            })

        result = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            if score < 0.5:
                continue

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            result.append([
                (left, top),(right, bottom),predicted_class, score
            ])

        image = lud.draw_box(image, result, outputs=saveto, datatype="coco")
        return image

    def close_session(self):
        self.sess.close()

def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = keras.backend.reshape(keras.backend.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = keras.backend.shape(feats)[1:3] # height, width
    grid_y = keras.backend.tile(keras.backend.reshape(keras.backend.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = keras.backend.tile(keras.backend.reshape(keras.backend.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = keras.backend.concatenate([grid_x, grid_y])
    grid = keras.backend.cast(grid, keras.backend.dtype(feats))

    feats = keras.backend.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (keras.backend.sigmoid(feats[..., :2]) + grid) / keras.backend.cast(grid_shape[::-1], keras.backend.dtype(feats))
    box_wh = keras.backend.exp(feats[..., 2:4]) * anchors_tensor / keras.backend.cast(input_shape[::-1], keras.backend.dtype(feats))
    box_confidence = keras.backend.sigmoid(feats[..., 4:5])
    box_class_probs = keras.backend.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = keras.backend.cast(input_shape, keras.backend.dtype(box_yx))
    image_shape = keras.backend.cast(image_shape, keras.backend.dtype(box_yx))
    new_shape = keras.backend.round(image_shape * keras.backend.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  keras.backend.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= keras.backend.concatenate([image_shape, image_shape])
    return boxes

def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
                                                                anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = keras.backend.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = keras.backend.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    input_shape = keras.backend.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = keras.backend.concatenate(boxes, axis=0)
    box_scores = keras.backend.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = keras.backend.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = keras.backend.gather(class_boxes, nms_index)
        class_box_scores = keras.backend.gather(class_box_scores, nms_index)
        classes = keras.backend.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = keras.backend.concatenate(boxes_, axis=0)
    scores_ = keras.backend.concatenate(scores_, axis=0)
    classes_ = keras.backend.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), PIL.Image.BICUBIC)
    new_image = PIL.Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image