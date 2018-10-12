import os
import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random

import lib_keras.maskrcnn.utils as mr_utils
import lib_keras.maskrcnn.dataset as mr_data
import lib_keras.maskrcnn.config as mr_config
import lib_keras.maskrcnn.net as mr_net

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

config = mr_config.BalloonConfig()
BALLOON_DIR = os.path.join("/input", "balloon")

dataset_train = mr_data.BalloonDataset()
dataset_train.load_balloon(BALLOON_DIR, "train")
dataset_train.prepare()

# Validation dataset
dataset_val = mr_data.BalloonDataset()
dataset_val.load_balloon(BALLOON_DIR, "val")
dataset_val.prepare()

# image_ids = np.random.choice(dataset.image_ids, 4)
# for image_id in image_ids:
#     image = dataset.load_image(image_id)
#     mask, class_ids = dataset.load_mask(image_id)
#     mr_utils.display_top_masks(image, mask, class_ids, dataset.class_names)

# image_id = random.choice(dataset.image_ids)
# image = dataset.load_image(image_id)
# mask, class_ids = dataset.load_mask(image_id)
# bbox = mr_utils.extract_bboxes(mask)
# print("image_id ", image_id, dataset.image_reference(image_id))
# mr_utils.log("image", image)
# mr_utils.log("mask", mask)
# mr_utils.log("class_ids", class_ids)
# mr_utils.log("bbox", bbox)
# # Display image and instances
# mr_utils.display_instances(image, bbox, mask, class_ids, dataset.class_names)


model = mr_net.MaskRCNN(mode="training", config=config,
                          model_dir="logs/")

init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    model.load_weights("utils/mask_rcnn_coco.h5", by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    model.load_weights(model.find_last(), by_name=True)

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=30,
            layers='heads')