import keras
import tensorflow as tf

def convert_result(output, anchors, nb_classes):
    """
    convert the model output into train label or test result comparable format.
    input:
    output: the output of model, for example, with shape, (batch_size, 13, 13, 425)
    anchors: the precomputed anchor size, list of tuples, every tuple indicates one width and height
    nb_classes: int, the number of prediction classes, to verify data integrity
    """
    anchors_length = keras.backend.shape(anchors)[0]
    output_shape = keras.backend.shape(output)

    tf_anchors = keras.backend.reshape(keras.backend.variable(anchors), [1, 1, 1, anchors_length, 2])

    #represent cx, cy
    height_index = keras.backend.arange(0,stop=output_shape[1])
    width_index  = keras.backend.arange(0,stop=output_shape[2])
    tmp1, tmp2 = tf.meshgrid(height_index, width_index)
    conv_index = tf.reshape(tf.concat([tmp1, tmp2], axis=0),(2, output_shape[1],output_shape[2]))
    conv_index = tf.transpose(conv_index, (1,2,0))
    conv_index = keras.backend.expand_dims(keras.backend.expand_dims(conv_index, 0),-2)#shape will be (1, 13, 13, 1, 2)
    conv_index = keras.backend.cast(conv_index, keras.backend.dtype(output))

    #reshape output
    output = keras.backend.reshape(output, [-1, output_shape[1], output_shape[2], anchors_length, nb_classes+5])

    #get sigmoid tx, ty, tw, th ,to in the paper
    bxy = keras.backend.sigmoid(output[...,:2])
    bwh = keras.backend.exp(output[...,2:4])
    to  = keras.backend.sigmoid(output[...,4:5])
    classes = keras.backend.softmax(output[...,5:])

    #use the equation to recover x,y, and get ratio
    dims = keras.backend.cast(keras.backend.reshape(output_shape[1:3],(1,1,1,1,2)), keras.backend.dtype(output))
    bxy = (bxy + conv_index)/dims
    bwh = bwh*tf_anchors/dims

    #the returned shape is (None, 13, 13, 5, 2), (None, 13, 13, 5, 2), (None, 13, 13, 5, 1), (None, 13, 13, 5, nb_clases), 5 is anchor length
    return bxy, bwh, to, classes

def filter_boxes(boxes, to, classes, to_threshold):
    """
    used to filter out boxes with confidence lower than to_threshold
    input:
    boxes: coordinates of all boxes
    to: confidence matrix
    classes: refer to result of convert_result
    to_threshold: threshold
    """
    confidence = to * classes
    #here refer to "Faster RCNN". For every point only keep points with maximum probability
    max_class = keras.backend.argmax(confidence, axis=-1)
    max_score = keras.backend.max(confidence, axis=-1)
    mask = max_score>=to_threshold #shape should be (None, 13, 13, 5)

    boxes = tf.boolean_mask(boxes, mask)
    to = tf.boolean_mask(max_score, mask)
    classes = tf.boolean_mask(max_class, mask)

    return boxes, to, classes

def draw_helper(result, image_size, max_boxes=10,to_threshold=0.6, iou_threshold=.5 ):
    """
    help to draw boxes in the image. That's to output boxes
    input:
    result: output by convert_result
    image_size:(height, width), help to calculate the relative position
    max_boxes: maximum number of boxes required by tf.non_max_suppression
    to_threshold: confidence threshold
    iou_threshold: used for non maximum supression
    """
    bxy, bwh, to, classes = result
    #convert bxy, bwh to top left and bottom right coordinates
    by_l = bxy[...,:1] - bwh[...,:1]/2.
    by_r = bxy[...,:1] + bwh[...,:1]/2.
    bx_l = bxy[...,1:] - bwh[...,1:]/2.
    bx_r = bxy[...,1:] + bwh[...,1:]/2.

    boxes = K.concatenate([bx_l,by_l, bx_r, by_r])
    #drop boxes with confidence lower than to_threshold

    boxes , to, classes = filter_boxes(boxes, to, classes, to_threshold)

    #scale back to image size
    height = image_size[0]
    width = image_size[1]
    image_dims = keras.backend.stack([height, width, height, width])
    image_dims = keras.backend.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims

    #get use non-maximum-supression
    max_boxes_tensor = keras.backend.variable(max_boxes, dtype='int32')
    keras.backend.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_index = tf.image.non_max_suppression(
        boxes, to, max_boxes_tensor, iou_threshold=iou_threshold)
    boxes = keras.backend.gather(boxes, nms_index)
    to = keras.backend.gather(to, nms_index)
    classes = keras.backend.gather(classes, nms_index)
    return boxes, to, classes