import matplotlib.pyplot as plt
import imgaug

import lib_keras.maskrcnn.utils as mr_utils
import lib_keras.maskrcnn.dataset as mr_data
import lib_keras.maskrcnn.config as mr_config
import lib_keras.maskrcnn.net as mr_net

PHASE = "train" # evaluate

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

config = mr_config.CocoConfig()

if PHASE == "train":

    dataset_train = mr_data.CocoDataset()
    dataset_train.load_coco("/input/COCO", "train", year=2014, auto_download=True)
    dataset_train.load_coco("/input/COCO", "valminusminival", year=2014, auto_download=True)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = mr_data.CocoDataset()
    val_type = "minival"
    dataset_val.load_coco("/input/COCO", val_type, year=2014, auto_download=True)
    dataset_val.prepare()

    # Image Augmentation
    # Right/Left flip 50% of the time
    augmentation = imgaug.augmenters.Fliplr(0.5)
    model = mr_net.MaskRCNN(mode="training", config=config,
                              model_dir="logs/")
    model.load_weights("utils/mask_rcnn_coco.h5", by_name=True)
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads',
                augmentation=augmentation)
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=120,
                layers='4+',
                augmentation=augmentation)
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=160,
                layers='all',
                augmentation=augmentation)
else:
    model = mr_net.MaskRCNN(mode="inference", config=config,
                              model_dir="logs/")
    model.load_weights("utils/mask_rcnn_coco.h5", by_name=True)
    # Validation dataset
    dataset_val = mr_data.CocoDataset()
    val_type = "minival"
    coco = dataset_val.load_coco("/input/COCO", val_type, year=2014, return_coco=True, auto_download=True)
    dataset_val.prepare()
    print("Running COCO evaluation on {} images.".format(500))
    mr_utils.evaluate_coco(model, dataset_val, coco, "bbox", limit=int(500))