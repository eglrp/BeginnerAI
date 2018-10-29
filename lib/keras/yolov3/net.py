import numpy as np
import keras
import functools
from functools import wraps

import lib.keras.yolov3.loss as k_loss

def create_model(CONFIG):
    input_shape = (CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])
    anchors = [float(x) for x in CONFIG["ANCHORS"].split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    num_classes = len(CONFIG["CLASSES"])
    load_pretrained = True
    freeze_body=2
    weights_path = CONFIG["MODEL_PATH"]

    keras.backend.clear_session() # get a new session
    image_input = keras.layers.Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [keras.layers.Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
                           num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = keras.layers.Lambda(k_loss.yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = keras.models.Model([model_body.input, *y_true], model_loss)

    return model

def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = keras.models.Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
        DarknetConv2D_BN_Leaky(256, (1,1)),
        keras.layers.UpSampling2D(2))(x)
    x = keras.layers.Concatenate()([x,darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
        DarknetConv2D_BN_Leaky(128, (1,1)),
        keras.layers.UpSampling2D(2))(x)
    x = keras.layers.Concatenate()([x,darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return keras.models.Model(inputs, [y1,y2,y3])

def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x

def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1,1)),
        DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
        DarknetConv2D_BN_Leaky(num_filters, (1,1)),
        DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
        DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
        DarknetConv2D(out_filters, (1,1)))(x)
    return x, y

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return functools.reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.1))

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = keras.layers.ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = keras.layers.Add()([x,y])
    return x

@wraps(keras.layers.Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': keras.regularizers.l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return keras.layers.Conv2D(*args, **darknet_conv_kwargs)

