import keras
import os
import tqdm
import PIL.Image as p_image
import numpy as np

import lib.keras.ssd.net as k_net
import lib.keras.ssd.loss as k_loss
import lib.utils.Config as k_config
import lib.utils.drawutils as draw

img_height = 300
img_width = 300

datatype = "voc"
DATA = k_config.TOTAL_CONFIG[datatype]

classes = ['background'] +  DATA["CLASSES"]

ssd_loss = k_loss.SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
model = keras.models.load_model("utils/keras_ssd_voc.h5",custom_objects={'AnchorBoxes': k_net.AnchorBoxes,
                                                      'L2Normalization': k_net.L2Normalization,
                                                      'DecodeDetections': k_net.DecodeDetections,
                                                      'compute_loss': ssd_loss.compute_loss})
confidence_threshold = 0.5
path = os.path.join("../testImages")
listfile = os.listdir(path)
for file in tqdm.tqdm(listfile):
    if file.endswith("jpg"):
        filename = file.split(".")[0]
        img_path = os.path.join(path, file)
        oriImage = p_image.open(img_path)

        img = keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
        img = keras.preprocessing.image.img_to_array(img)
        input_images = np.array([img])

        y_pred = model.predict(input_images)

        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

        np.set_printoptions(precision=2, suppress=True, linewidth=90)

        result = []

        for box in y_pred_thresh[0]:
            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
            xmin = box[2] * oriImage.width / img_width
            ymin = box[3] * oriImage.height / img_height
            xmax = box[4] * oriImage.width / img_width
            ymax = box[5] * oriImage.height / img_height
            result.append([(xmin, ymin),(xmax,ymax),classes[int(box[0])], box[1]])
        draw.draw_box(oriImage, result, outputs="outputs/%s.jpg" % filename)

