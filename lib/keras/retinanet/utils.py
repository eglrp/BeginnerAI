import numpy as np
import keras
import tensorflow as tf
import PIL
import cv2
import six

class AnchorParameters:
    """ The parameteres that define how anchors are generated.

    Args
        sizes   : List of sizes to use. Each size corresponds to one feature level.
        strides : List of strides to use. Each stride correspond to one feature level.
        ratios  : List of ratios to use per location in a feature map.
        scales  : List of scales to use per location in a feature map.
    """
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes   = sizes
        self.strides = strides
        self.ratios  = ratios
        self.scales  = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


"""
The default anchor parameters.
"""
AnchorParameters.default = AnchorParameters(
    sizes   = [32, 64, 128, 256, 512],
    strides = [8, 16, 32, 64, 128],
    ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
    scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)

def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = AnchorParameters.default.ratios

    if scales is None:
        scales = AnchorParameters.default.scales

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors

def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """
    shift_x = (keras.backend.arange(0, shape[1], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride
    shift_y = (keras.backend.arange(0, shape[0], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride

    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = keras.backend.reshape(shift_x, [-1])
    shift_y = keras.backend.reshape(shift_y, [-1])

    shifts = keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts            = keras.backend.transpose(shifts)
    number_of_anchors = keras.backend.shape(anchors)[0]

    k = keras.backend.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + \
                      keras.backend.cast(keras.backend.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors

def resize_images(images, size, method='bilinear', align_corners=False):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/image/resize_images .

    Args
        method: The method used for interpolation. One of ('bilinear', 'nearest', 'bicubic', 'area').
    """
    methods = {
        'bilinear': tf.image.ResizeMethod.BILINEAR,
        'nearest' : tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bicubic' : tf.image.ResizeMethod.BICUBIC,
        'area'    : tf.image.ResizeMethod.AREA,
    }
    return tf.image.resize_images(images, size, methods[method], align_corners)

def bbox_transform_inv(boxes, deltas, mean=None, std=None):
    """ Applies deltas (usually regression results) to boxes (usually anchors).

    Before applying the deltas to the boxes, the normalization that was previously applied (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator. They are unnormalized in this function and then applied to the boxes.

    Args
        boxes : np.array of shape (B, N, 4), where B is the batch size, N the number of boxes and 4 values for (x1, y1, x2, y2).
        deltas: np.array of same shape as boxes. These deltas (d_x1, d_y1, d_x2, d_y2) are a factor of the width/height.
        mean  : The mean value used when computing deltas (defaults to [0, 0, 0, 0]).
        std   : The standard deviation used when computing deltas (defaults to [0.2, 0.2, 0.2, 0.2]).

    Returns
        A np.array of the same shape as boxes, but with deltas applied to each box.
        The mean and std are used during training to normalize the regression values (networks love normalization).
    """
    if mean is None:
        mean = [0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2]

    width  = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    x1 = boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
    y1 = boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
    x2 = boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
    y2 = boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height

    pred_boxes = keras.backend.stack([x1, y1, x2, y2], axis=2)

    return pred_boxes

def filter_detections(
        boxes,
        classification,
        other                 = [],
        class_specific_filter = True,
        nms                   = True,
        score_threshold       = 0.05,
        max_detections        = 300,
        nms_threshold         = 0.5
):
    """ Filter detections using the boxes and classification values.

    Args
        boxes                 : Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification        : Tensor of shape (num_boxes, num_classes) containing the classification scores.
        other                 : List of tensors of shape (num_boxes, ...) to filter along with the boxes and classification scores.
        class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
        nms                   : Flag to enable/disable non maximum suppression.
        score_threshold       : Threshold used to prefilter the boxes with.
        max_detections        : Maximum number of detections to keep.
        nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.

    Returns
        A list of [boxes, scores, labels, other[0], other[1], ...].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """
    def _filter_detections(scores, labels):
        # threshold based on score
        indices = tf.where(keras.backend.greater(scores, score_threshold))

        if nms:
            filtered_boxes  = tf.gather_nd(boxes, indices)
            filtered_scores = keras.backend.gather(scores, indices)[:, 0]

            # perform NMS
            nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections, iou_threshold=nms_threshold)

            # filter indices based on NMS
            indices = keras.backend.gather(indices, nms_indices)

        # add indices to list of all indices
        labels = tf.gather_nd(labels, indices)
        indices = keras.backend.stack([indices[:, 0], labels], axis=1)

        return indices

    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * tf.ones((keras.backend.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
        indices = keras.backend.concatenate(all_indices, axis=0)
    else:
        scores  = keras.backend.max(classification, axis    = 1)
        labels  = keras.backend.argmax(classification, axis = 1)
        indices = _filter_detections(scores, labels)

    # select top k
    scores              = tf.gather_nd(classification, indices)
    labels              = indices[:, 1]
    scores, top_indices = tf.nn.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))

    # filter input using the final set of indices
    indices             = keras.backend.gather(indices[:, 0], top_indices)
    boxes               = keras.backend.gather(boxes, indices)
    labels              = keras.backend.gather(labels, top_indices)
    other_              = [keras.backend.gather(o, indices) for o in other]

    # zero pad the outputs
    pad_size = keras.backend.maximum(0, max_detections - keras.backend.shape(scores)[0])
    boxes    = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores   = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels   = tf.pad(labels, [[0, pad_size]], constant_values=-1)
    labels   = keras.backend.cast(labels, 'int32')
    other_   = [tf.pad(o, [[0, pad_size]] + [[0, 0] for _ in range(1, len(o.shape))], constant_values=-1) for o in other_]

    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    for o, s in zip(other_, [list(keras.backend.int_shape(o)) for o in other]):
        o.set_shape([max_detections] + s[1:])

    return [boxes, scores, labels] + other_

def preprocess_image(x, mode='caffe'):
    """ Preprocess an image by subtracting the ImageNet mean.

    Args
        x: np.array of shape (None, None, 3) or (3, None, None).
        mode: One of "caffe" or "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.

    Returns
        The input with the ImageNet mean subtracted.
    """
    # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already
    x = x.astype(keras.backend.floatx())
    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x

def read_image_bgr(path):
    """ Read an image in BGR format.

    Args
        path: Path to the image.
    """
    image = np.asarray(PIL.Image.open(path).convert('RGB'))
    return image[:, :, ::-1].copy()

def resize_image(img, min_side=800, max_side=1333):
    """ Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    # compute scale to resize the image
    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale

def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    """ Compute an image scale such that the image size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resizing scale.
    """
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale

def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)

def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def anchor_targets_bbox(
        anchors,
        image_group,
        annotations_group,
        num_classes,
        negative_overlap=0.4,
        positive_overlap=0.5
):
    """ Generate anchor targets for bbox detection.

    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        image_group: List of BGR images.
        annotations_group: List of annotations (np.array of shape (N, 5) for (x1, y1, x2, y2, label)).
        num_classes: Number of classes to predict.
        mask_shape: If the image is padded with zeros, mask_shape can be used to mark the relevant part of the image.
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns
        labels_batch: batch that contains labels & anchor states (np.array of shape (batch_size, N, num_classes + 1),
                      where N is the number of anchors for an image and the last column defines the anchor state (-1 for ignore, 0 for bg, 1 for fg).
        regression_batch: batch that contains bounding-box regression targets for an image & anchor states (np.array of shape (batch_size, N, 4 + 1),
                      where N is the number of anchors for an image, the first 4 columns define regression targets for (x1, y1, x2, y2) and the
                      last column defines anchor states (-1 for ignore, 0 for bg, 1 for fg).
    """

    assert(len(image_group) == len(annotations_group)), "The length of the images and annotations need to be equal."
    assert(len(annotations_group) > 0), "No data received to compute anchor targets for."
    for annotations in annotations_group:
        assert('bboxes' in annotations), "Annotations should contain bboxes."
        assert('labels' in annotations), "Annotations should contain labels."

    batch_size = len(image_group)

    regression_batch  = np.zeros((batch_size, anchors.shape[0], 4 + 1), dtype=keras.backend.floatx())
    labels_batch      = np.zeros((batch_size, anchors.shape[0], num_classes + 1), dtype=keras.backend.floatx())

    # compute labels and regression targets
    for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
        if annotations['bboxes'].shape[0]:
            # obtain indices of gt annotations with the greatest overlap
            positive_indices, ignore_indices, argmax_overlaps_inds = compute_gt_annotations(anchors, annotations['bboxes'], negative_overlap, positive_overlap)

            labels_batch[index, ignore_indices, -1]       = -1
            labels_batch[index, positive_indices, -1]     = 1

            regression_batch[index, ignore_indices, -1]   = -1
            regression_batch[index, positive_indices, -1] = 1

            # compute target class labels
            labels_batch[index, positive_indices, annotations['labels'][argmax_overlaps_inds[positive_indices]].astype(int)] = 1

            regression_batch[index, :, :-1] = bbox_transform(anchors, annotations['bboxes'][argmax_overlaps_inds, :])

        # ignore annotations outside of image
        if image.shape:
            anchors_centers = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
            indices = np.logical_or(anchors_centers[:, 0] >= image.shape[1], anchors_centers[:, 1] >= image.shape[0])

            labels_batch[index, indices, -1]     = -1
            regression_batch[index, indices, -1] = -1

    return regression_batch, labels_batch

def compute_gt_annotations(
        anchors,
        annotations,
        negative_overlap=0.4,
        positive_overlap=0.5
):
    """ Obtain indices of gt annotations with the greatest overlap.

    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        annotations: np.array of shape (N, 5) for (x1, y1, x2, y2, label).
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns
        positive_indices: indices of positive anchors
        ignore_indices: indices of ignored anchors
        argmax_overlaps_inds: ordered overlaps indices
    """

    overlaps = compute_overlap(anchors.astype(np.float64), annotations.astype(np.float64))
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

    # assign "dont care" labels
    positive_indices = max_overlaps >= positive_overlap
    ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices

    return positive_indices, ignore_indices, argmax_overlaps_inds

def bbox_transform(anchors, gt_boxes, mean=None, std=None):
    """Compute bounding-box regression targets for an image."""

    if mean is None:
        mean = np.array([0, 0, 0, 0])
    if std is None:
        std = np.array([0.2, 0.2, 0.2, 0.2])

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    anchor_widths  = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
    targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
    targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
    targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights

    targets = np.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2))
    targets = targets.T

    targets = (targets - mean) / std

    return targets

def compute_overlap(boxes,query_boxes):
    """
    Args
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        box_area = (
                (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
                (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                    min(boxes[n, 2], query_boxes[k, 2]) -
                    max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                        min(boxes[n, 3], query_boxes[k, 3]) -
                        max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def guess_shapes(image_shape, pyramid_levels):
    """Guess shapes based on pyramid levels.

    Args
         image_shape: The shape of the image.
         pyramid_levels: A list of what pyramid levels are used.

    Returns
        A list of image shapes at each pyramid level.
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes

def adjust_transform_for_image(transform, image, relative_translation):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    height, width, channels = image.shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result

def translation(translation):
    """ Construct a homogeneous 2D translation matrix.
    # Arguments
        translation: the translation 2D vector
    # Returns
        the translation matrix as 3 by 3 numpy array
    """
    return np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])

def change_transform_origin(transform, center):
    """ Create a new transform representing the same transformation,
        only with the origin of the linear part changed.
    Args
        transform: the transformation matrix
        center: the new origin of the transformation
    Returns
        translate(center) * transform * translate(-center)
    """
    center = np.array(center)
    return np.linalg.multi_dot([translation(center), transform, translation(-center)])

def random_transform(
        min_rotation=0,
        max_rotation=0,
        min_translation=(0, 0),
        max_translation=(0, 0),
        min_shear=0,
        max_shear=0,
        min_scaling=(1, 1),
        max_scaling=(1, 1),
        flip_x_chance=0,
        flip_y_chance=0
):
    """ Create a random transformation.

    The transformation consists of the following operations in this order (from left to right):
      * rotation
      * translation
      * shear
      * scaling
      * flip x (if applied)
      * flip y (if applied)

    Note that by default, the data generators in `keras_retinanet.preprocessing.generators` interpret the translation
    as factor of the image size. So an X translation of 0.1 would translate the image by 10% of it's width.
    Set `relative_translation` to `False` in the `TransformParameters` of a data generator to have it interpret
    the translation directly as pixel distances instead.

    Args
        min_rotation:    The minimum rotation in radians for the transform as scalar.
        max_rotation:    The maximum rotation in radians for the transform as scalar.
        min_translation: The minimum translation for the transform as 2D column vector.
        max_translation: The maximum translation for the transform as 2D column vector.
        min_shear:       The minimum shear angle for the transform in radians.
        max_shear:       The maximum shear angle for the transform in radians.
        min_scaling:     The minimum scaling for the transform as 2D column vector.
        max_scaling:     The maximum scaling for the transform as 2D column vector.
        flip_x_chance:   The chance (0 to 1) that a transform will contain a flip along X direction.
        flip_y_chance:   The chance (0 to 1) that a transform will contain a flip along Y direction.
        prng:            The pseudo-random number generator to use.
    """
    return np.linalg.multi_dot([
        random_rotation(min_rotation, max_rotation),
        random_translation(min_translation, max_translation),
        random_shear(min_shear, max_shear),
        random_scaling(min_scaling, max_scaling),
        random_flip(flip_x_chance, flip_y_chance)
    ])

def _random_vector(min, max):
    """ Construct a random vector between min and max.
    Args
        min: the minimum value for each component
        max: the maximum value for each component
    """
    min = np.array(min)
    max = np.array(max)
    assert min.shape == max.shape
    assert len(min.shape) == 1
    return np.random.uniform(min, max)

def scaling(factor):
    """ Construct a homogeneous 2D scaling matrix.
    Args
        factor: a 2D vector for X and Y scaling
    Returns
        the zoom matrix as 3 by 3 numpy array
    """
    return np.array([
        [factor[0], 0, 0],
        [0, factor[1], 0],
        [0, 0, 1]
    ])

def random_scaling(min, max):
    """ Construct a random 2D scale matrix between -max and max.
    Args
        min:  a 2D vector containing the minimum scaling factor for X and Y.
        min:  a 2D vector containing The maximum scaling factor for X and Y.
        prng: the pseudo-random number generator to use.
    Returns
        a homogeneous 3 by 3 scaling matrix
    """
    return scaling(_random_vector(min, max))


def random_flip(flip_x_chance, flip_y_chance):
    """ Construct a transformation randomly containing X/Y flips (or not).
    Args
        flip_x_chance: The chance that the result will contain a flip along the X axis.
        flip_y_chance: The chance that the result will contain a flip along the Y axis.
        prng:          The pseudo-random number generator to use.
    Returns
        a homogeneous 3 by 3 transformation matrix
    """
    flip_x = np.random.uniform(0, 1) < flip_x_chance
    flip_y = np.random.uniform(0, 1) < flip_y_chance
    # 1 - 2 * bool gives 1 for False and -1 for True.
    return scaling((1 - 2 * flip_x, 1 - 2 * flip_y))

def random_rotation(min, max):
    """ Construct a random rotation between -max and max.
    Args
        min:  a scalar for the minimum absolute angle in radians
        max:  a scalar for the maximum absolute angle in radians
        prng: the pseudo-random number generator to use.
    Returns
        a homogeneous 3 by 3 rotation matrix
    """
    return rotation(np.random.uniform(min, max))

def rotation(angle):
    """ Construct a homogeneous 2D rotation matrix.
    Args
        angle: the angle in radians
    Returns
        the rotation matrix as 3 by 3 numpy array
    """
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])

def random_translation(min, max):
    """ Construct a random 2D translation between min and max.
    Args
        min:  a 2D vector with the minimum translation for each dimension
        max:  a 2D vector with the maximum translation for each dimension
        prng: the pseudo-random number generator to use.
    Returns
        a homogeneous 3 by 3 translation matrix
    """
    return translation(_random_vector(min, max))

def translation(translation):
    """ Construct a homogeneous 2D translation matrix.
    # Arguments
        translation: the translation 2D vector
    # Returns
        the translation matrix as 3 by 3 numpy array
    """
    return np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])

def random_shear(min, max):
    """ Construct a random 2D shear matrix with shear angle between -max and max.
    Args
        min:  the minimum shear angle in radians.
        max:  the maximum shear angle in radians.
        prng: the pseudo-random number generator to use.
    Returns
        a homogeneous 3 by 3 shear matrix
    """
    return shear(np.random.uniform(min, max))

def shear(angle):
    """ Construct a homogeneous 2D shear matrix.
    Args
        angle: the shear angle in radians
    Returns
        the shear matrix as 3 by 3 numpy array
    """
    return np.array([
        [1, -np.sin(angle), 0],
        [0,  np.cos(angle), 0],
        [0, 0, 1]
    ])


def random_transform_generator(prng=None, **kwargs):
    """ Create a random transform generator.

    Uses a dedicated, newly created, properly seeded PRNG by default instead of the global DEFAULT_PRNG.

    The transformation consists of the following operations in this order (from left to right):
      * rotation
      * translation
      * shear
      * scaling
      * flip x (if applied)
      * flip y (if applied)

    Note that by default, the data generators in `keras_retinanet.preprocessing.generators` interpret the translation
    as factor of the image size. So an X translation of 0.1 would translate the image by 10% of it's width.
    Set `relative_translation` to `False` in the `TransformParameters` of a data generator to have it interpret
    the translation directly as pixel distances instead.

    Args
        min_rotation:    The minimum rotation in radians for the transform as scalar.
        max_rotation:    The maximum rotation in radians for the transform as scalar.
        min_translation: The minimum translation for the transform as 2D column vector.
        max_translation: The maximum translation for the transform as 2D column vector.
        min_shear:       The minimum shear angle for the transform in radians.
        max_shear:       The maximum shear angle for the transform in radians.
        min_scaling:     The minimum scaling for the transform as 2D column vector.
        max_scaling:     The maximum scaling for the transform as 2D column vector.
        flip_x_chance:   The chance (0 to 1) that a transform will contain a flip along X direction.
        flip_y_chance:   The chance (0 to 1) that a transform will contain a flip along Y direction.
        prng:            The pseudo-random number generator to use.
    """

    if prng is None:
        # RandomState automatically seeds using the best available method.
        prng = np.random.RandomState()

    while True:
        yield random_transform(**kwargs)

def apply_transform(matrix, image, params):
    """
    Apply a transformation to an image.

    The origin of transformation is at the top left corner of the image.

    The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
    Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.

    Args
      matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
      image:  The image to transform.
      params: The transform parameters (see TransformParameters)
    """
    output = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize       = (image.shape[1], image.shape[0]),
        flags       = params.cvInterpolation(),
        borderMode  = params.cvBorderMode(),
        borderValue = params.cval,
    )
    return output

def transform_aabb(transform, aabb):
    """ Apply a transformation to an axis aligned bounding box.

    The result is a new AABB in the same coordinate system as the original AABB.
    The new AABB contains all corner points of the original AABB after applying the given transformation.

    Args
        transform: The transformation to apply.
        x1:        The minimum x value of the AABB.
        y1:        The minimum y value of the AABB.
        x2:        The maximum x value of the AABB.
        y2:        The maximum y value of the AABB.
    Returns
        The new AABB as tuple (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = aabb
    # Transform all 4 corners of the AABB.
    points = transform.dot([
        [x1, x2, x1, x2],
        [y1, y2, y2, y1],
        [1,  1,  1,  1 ],
    ])

    # Extract the min and max corners again.
    min_corner = points.min(axis=1)
    max_corner = points.max(axis=1)

    return [min_corner[0], min_corner[1], max_corner[0], max_corner[1]]

def parse_anchor_parameters(config):
    ratios  = np.array(list(map(float, config['anchor_parameters']['ratios'].split(' '))), keras.backend.floatx())
    scales  = np.array(list(map(float, config['anchor_parameters']['scales'].split(' '))), keras.backend.floatx())
    sizes   = list(map(int, config['anchor_parameters']['sizes'].split(' ')))
    strides = list(map(int, config['anchor_parameters']['strides'].split(' ')))

    return AnchorParameters(sizes, strides, ratios, scales)

def anchors_for_shape(
        image_shape,
        pyramid_levels=None,
        anchor_params=None,
        shapes_callback=None,
):
    """ Generators anchors for a given shape.

    Args
        image_shape: The shape of the image.
        pyramid_levels: List of ints representing which pyramids to use (defaults to [3, 4, 5, 6, 7]).
        anchor_params: Struct containing anchor parameters. If None, default values are used.
        shapes_callback: Function to call for getting the shape of the image at different pyramid levels.

    Returns
        np.array of shape (N, 4) containing the (x1, y1, x2, y2) coordinates for the anchors.
    """

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    if anchor_params is None:
        anchor_params = AnchorParameters.default

    if shapes_callback is None:
        shapes_callback = guess_shapes
    image_shapes = shapes_callback(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales
        )
        shifted_anchors = anchors_shift(image_shapes[idx], anchor_params.strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors

def anchors_shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """

    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors

def _findNode(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            six.raise_from(ValueError('illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result

def assert_training_model(model):
    """ Assert that the model is a training model.
    """
    assert(all(output in model.output_names for output in ['regression', 'classification'])), \
        "Input is not a training model (no 'regression' and 'classification' outputs were found, outputs are: {}).".format(model.output_names)

