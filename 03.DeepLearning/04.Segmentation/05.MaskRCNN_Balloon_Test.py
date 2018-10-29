import lib.keras.maskrcnn.config as mr_config
import lib.keras.maskrcnn.net as mr_net
import random
import lib.keras.maskrcnn.dataset as mr_data
import lib.keras.maskrcnn.utils as mr_utils
import skimage
import numpy as np

class InferenceConfig(mr_config.BalloonConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = mr_net.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir="logs/")

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = "utils/mask_rcnn_balloon.h5" #model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Read image
image = skimage.io.imread("../testImages/12.jpg")
# Detect objects
r = model.detect([image], verbose=1)[0]
# Color splash
splash = color_splash(image, r['masks'])
# Save output
skimage.io.imsave("output/2.jpg", splash)