'''
@author: JJZHK
@license: (C) Copyright 2017-2023, Node Supply Chain Manager Corporation Limited.
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: Keras_SSD.py
@time: 2018/10/26 09:52
@desc: 
'''
import keras
import os
import math

import lib.keras.ssd.net as k_net
import lib.utils.Config as k_config
import lib.keras.ssd.loss as k_loss
import lib.keras.ssd.data as k_data
import lib.utils.image as lui

datatype = "voc"
DATA = k_config.TOTAL_CONFIG[datatype]

classes = DATA["CLASSES"]

CONFIG = {
    "IMAGE_SIZE" : 300,
    "IMAGE_CHANNEL" : 3,
    "CLASSES" : classes,
    "SCALES" : [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],# The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
    "ASPECT_RATIOS" : [[1.0, 2.0, 0.5],
                       [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                       [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                       [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                       [1.0, 2.0, 0.5],
                       [1.0, 2.0, 0.5]],# The anchor box aspect ratios used in the original SSD300; the order matters
    "TWO_BOXES_FOR_AR1" : True,
    "STEPS" : [8, 16, 32, 64, 100, 300],# The space between two adjacent anchor box center points for each predictor layer.
    "OFFSETS" : [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],# The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
    "CLIP_BOXES" : False,# Whether or not to clip the anchor boxes to lie entirely within the image boundaries
    "VARIANCES" : [0.1, 0.1, 0.2, 0.2],# The variances by which the encoded target coordinates are divided as in the original implementation
    "NORMALIZE_COORDS" : True,
    "MEAN_COLOR" : [123, 117, 104],# The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
    "SWAP_CHANNELS" : [2, 1, 0],# The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
    "MODEL" : "training",
    "WEIGHTS_FILE" : "utils/keras_ssd_voc_weights.h5",
    "DATA_ROOT" : "/input/voc"
}

def lr_schedule(epoch):
    if epoch < 80:
        return 0.001
    elif epoch < 100:
        return 0.0001
    else:
        return 0.00001

keras.backend.clear_session()
model = k_net.SSD_300(image_size=(CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_CHANNEL"]),
                      n_classes=20,
                      mode='training',
                      l2_regularization=0.0005,
                      scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                      aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                               [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                               [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                               [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                               [1.0, 2.0, 0.5],
                                               [1.0, 2.0, 0.5]],
                      two_boxes_for_ar1=True,
                      steps=[8, 16, 32, 64, 100, 300],
                      offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                      clip_boxes=False,
                      variances=[0.1, 0.1, 0.2, 0.2],
                      normalize_coords=True,
                      subtract_mean=[123, 117, 104],
                      swap_channels=[2, 1, 0])
model.load_weights(CONFIG["WEIGHTS_FILE"], by_name=True)

sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
ssd_loss = k_loss.SSDLoss(neg_pos_ratio=3, alpha=1.0)
model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)

train_dataset = k_data.DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = k_data.DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

b_class = ['background'] + classes

train_dataset.parse_xml(images_dirs=[os.path.join(CONFIG["DATA_ROOT"], "JPEGImages")],
                        image_set_filenames=[os.path.join(CONFIG["DATA_ROOT"], "MainSet", "det_train.txt")],
                        annotations_dirs=[os.path.join(CONFIG["DATA_ROOT"], "Annotations")],
                        classes=b_class,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)

val_dataset.parse_xml(images_dirs=[os.path.join(CONFIG["DATA_ROOT"], "JPEGImages")],
                      image_set_filenames=[os.path.join(CONFIG["DATA_ROOT"], "MainSet", "det_test.txt")],
                      annotations_dirs=[os.path.join(CONFIG["DATA_ROOT"], "Annotations")],
                      classes=b_class,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=True,
                      ret=False)

batch_size = 32 # Change the batch size if you like, or if you run into GPU memory issues.

# 4: Set the image transformations for pre-processing and data augmentation options.

# For the training generator:
ssd_data_augmentation = k_data.SSDDataAugmentation(img_height=CONFIG["IMAGE_SIZE"],
                                            img_width=CONFIG["IMAGE_SIZE"],
                                            background=CONFIG["MEAN_COLOR"])

# For the validation generator:
convert_to_3_channels = lui.ConvertTo3Channels()
resize = lui.Resize(height=CONFIG["IMAGE_SIZE"], width=CONFIG["IMAGE_SIZE"])

# 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

ssd_input_encoder = k_data.SSDInputEncoder(img_height=CONFIG["IMAGE_SIZE"],
                                           img_width=CONFIG["IMAGE_SIZE"],
                                           n_classes=20,
                                           predictor_sizes=predictor_sizes,
                                           scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                                           aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                                    [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                                    [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                                    [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                                    [1.0, 2.0, 0.5],
                                                                    [1.0, 2.0, 0.5]],
                                           two_boxes_for_ar1=True,
                                           steps=[8, 16, 32, 64, 100, 300],
                                           offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                           clip_boxes=False,
                                           variances=[0.1, 0.1, 0.2, 0.2],
                                           matching_type='multi',
                                           pos_iou_threshold=0.5,
                                           neg_iou_limit=0.5,
                                           normalize_coords=True)

# 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

# Define model callbacks.

# TODO: Set the filepath under which you want to save the model.
model_checkpoint = keras.callbacks.ModelCheckpoint(filepath='ssd300_pascal_07+12_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto',
                                   period=1)
#model_checkpoint.best =

csv_logger = keras.callbacks.CSVLogger(filename='ssd300_pascal_07+12_training_log.csv',
                       separator=',',
                       append=True)

learning_rate_scheduler = keras.callbacks.LearningRateScheduler(schedule=lr_schedule,
                                                verbose=1)

terminate_on_nan = keras.callbacks.TerminateOnNaN()

callbacks = [model_checkpoint,
             csv_logger,
             learning_rate_scheduler,
             terminate_on_nan]

# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch   = 0
final_epoch     = 120
steps_per_epoch = 1000

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              validation_data=val_generator,
                              validation_steps=math.ceil(val_dataset_size/batch_size),
                              initial_epoch=initial_epoch)
