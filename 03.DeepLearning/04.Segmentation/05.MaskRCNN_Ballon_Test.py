import lib_keras.maskrcnn.config as mr_config
import lib_keras.maskrcnn.net as mr_net
import random
import lib_keras.maskrcnn.dataset as mr_data
import lib_keras.maskrcnn.utils as mr_utils

class InferenceConfig(mr_config.ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = mr_net.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir="logs/")

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = "utils/mask_rcnn_shape.h5" #model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

config = mr_config.ShapesConfig()

dataset_train = mr_data.ShapesDataset()
dataset_train.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

dataset_val = mr_data.ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    mr_utils.load_image_gt(dataset_val, inference_config,
                           image_id, use_mini_mask=False)

mr_utils.log("original_image", original_image)
mr_utils.log("image_meta", image_meta)
mr_utils.log("gt_class_id", gt_class_id)
mr_utils.log("gt_bbox", gt_bbox)
mr_utils.log("gt_mask", gt_mask)

mr_utils.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8, 8))