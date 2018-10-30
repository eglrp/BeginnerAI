# https://github.com/yhcc/yolo2
import lib.keras.yolov2.net as k_net
import keras
import tensorflow.keras as k

CONFIG = {
    "MODEL_PATH" : "utils/keras_yolov2.h5",
    "ANCHORS" : "0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828",
    "CLASSES" : ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'],
    "SAMPLES_PER_EPOCH":16,
    "EPOCHS" : 100,
    "NB_VAL_SAMPLES" : None,
    "SAVE_PATH" : "outputs/new_keras_yolov2.h5",
    "BATCH_SIZE" : 16,
    "IMAGE_SIZE" : 416,
    "FILE_PATH" : "utils/keras_yolov2_train.txt"

}

model = k_net.model_to_train(CONFIG, keras.optimizers.SGD(1e-6, momentum=0.9))
k_net.train_model(model, CONFIG)