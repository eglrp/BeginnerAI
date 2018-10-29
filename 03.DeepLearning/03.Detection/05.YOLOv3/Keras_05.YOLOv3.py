import keras
import lib.keras.yolov3.net as k_net
import lib.keras.yolov3.data as k_data
import numpy as np

keras.models.load_model("utils/yolo.h5")
'''
Model_path使用keras_yolov3_coco.h5和keras_yolov3_darknet53_pre.h5都可以。
'''

CONFIG = {
    "DATA_FILE" : "utils/keras_yolov3_train.txt",
    "MODEL_PATH" : "utils/keras_yolov3_darknet53_pre.h5",
    "IMAGE_SIZE" : 416,
    "IMAGE_CHANNEL" : 3,
    "CLASSES" : ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'],
    "ANCHORS" : "10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326",
    "LOG_DIR" : "logs/"
}
input_shape = (CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])
anchors = [float(x) for x in CONFIG["ANCHORS"].split(',')]
anchors = np.array(anchors).reshape(-1, 2)
num_classes = len(CONFIG["CLASSES"])

model = k_net.create_model(CONFIG)
logging = keras.callbacks.TensorBoard(log_dir=CONFIG["LOG_DIR"])
checkpoint = keras.callbacks.ModelCheckpoint(CONFIG["LOG_DIR"] + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                             monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

val_split = 0.1
with open(CONFIG["DATA_FILE"]) as f:
    lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val

model.compile(optimizer=keras.optimizers.Adam(lr=1e-3), loss={
    # use custom yolo_loss Lambda layer.
    'yolo_loss': lambda y_true, y_pred: y_pred})

batch_size = 32
print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
model.fit_generator(k_data.data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                    steps_per_epoch=max(1, num_train//batch_size),
                    validation_data=k_data.data_generator_wrapper(lines[num_train:], batch_size,
                                                                  input_shape, anchors, num_classes),
                    validation_steps=max(1, num_val//batch_size),
                    epochs=50,
                    initial_epoch=0,
                    callbacks=[logging, checkpoint])
model.save_weights(CONFIG["LOG_DIR"] + 'trained_weights_stage_1.h5')

for i in range(len(model.layers)):
    model.layers[i].trainable = True
model.compile(optimizer=keras.optimizers.Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
print('Unfreeze all of the layers.')

batch_size = 32 # note that more GPU memory is required after unfreezing the body
print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
model.fit_generator(k_data.data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                    steps_per_epoch=max(1, num_train//batch_size),
                    validation_data=k_data.data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                    validation_steps=max(1, num_val//batch_size),
                    epochs=100,
                    initial_epoch=50,
                    callbacks=[logging, checkpoint, reduce_lr, early_stopping])
model.save_weights(CONFIG["LOG_DIR"] + 'trained_weights_final.h5')


