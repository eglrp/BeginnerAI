import keras
import tensorflow as tf
import lib_keras.retinanet.data as k_data
import lib_keras.retinanet.net as k_net
import lib_keras.retinanet.utils as k_utils
import lib_keras.retinanet.loss as k_loss

def create_models(backbone_retinanet, num_classes, weights, multi_gpu=0, freeze_backbone=False, config=None):
    """ Creates three models (model, training_model, prediction_model).

    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    modifier = None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors   = None
    if config and 'anchor_parameters' in config:
        anchor_params = k_utils.parse_anchor_parameters(config)
        num_anchors   = anchor_params.num_anchors()

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier)
        training_model = model

    # make prediction model
    prediction_model = k_net.retinanet_bbox(model=model, anchor_params=anchor_params)

    # compile model
    training_model.compile(
        loss={
            'regression'    : k_loss.smooth_l1(),
            'classification': k_loss.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )

    return model, training_model, prediction_model

def create_callbacks(model, training_model, prediction_model, validation_generator):
    """ Creates the callbacks to use during training.

    Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir                = "logs/",
        histogram_freq         = 0,
        batch_size             = 1,
        write_graph            = True,
        write_grads            = False,
        write_images           = False,
        embeddings_freq        = 0,
        embeddings_layer_names = None,
        embeddings_metadata    = None
    )
    callbacks.append(tensorboard_callback)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = 'loss',
        factor     = 0.1,
        patience   = 2,
        verbose    = 1,
        mode       = 'auto',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 0
    ))

    return callbacks

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(session)

model = k_net.ResNetBackbone("resnet50")

common_args = {
    'batch_size'       : 1,
    'config'           : None,
    'image_min_side'   : 800,
    'image_max_side'   : 1333,
    'preprocess_image' : model.preprocess_image,
}

train_generator = k_data.PascalVocGenerator("/input/voc", "det_train", **common_args)
test_generator = k_data.PascalVocGenerator("/input/voc", "det_test", **common_args)


print('Creating model, this may take a second...')
model, training_model, prediction_model = create_models(
    backbone_retinanet=model.retinanet,
    num_classes=train_generator.num_classes(),
    weights=None,
    multi_gpu=0,
    freeze_backbone=None,
    config=None
)

print(model.summary())

callbacks = create_callbacks(
    model,
    training_model,
    prediction_model,
    test_generator
)

# start training
training_model.fit_generator(
    generator=train_generator,
    steps_per_epoch=10000,
    epochs=50,
    verbose=1,
    callbacks=callbacks)

training_model.save("outputs/keras_retinanet_voc.h5")