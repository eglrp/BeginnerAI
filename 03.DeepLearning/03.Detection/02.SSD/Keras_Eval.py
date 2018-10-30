'''
@author: JJZHK
@license: (C) Copyright 2017-2023, Node Supply Chain Manager Corporation Limited.
@contact: jeffcobile@gmail.com
@Software : PyCharm
@file: Keras_Eval.py
@time: 2018/10/30 13:35
@desc: 
'''
import keras
import os
import tqdm
import PIL.Image as p_image
import numpy as np
import lib.keras.ssd.data as k_data
import lib.keras.ssd.net as k_net
import lib.keras.ssd.loss as k_loss
import lib.utils.Config as k_config
import lib.keras.ssd.eval as k_eval
import matplotlib.pyplot as plt

import lib.utils.drawutils as draw

img_height = 300
img_width = 300

datatype = "voc"
DATA = k_config.TOTAL_CONFIG[datatype]

classes = ['background'] +  DATA["CLASSES"]
keras.backend.clear_session()
ssd_loss = k_loss.SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
model = keras.models.load_model("utils/keras_ssd_voc.h5",custom_objects={'AnchorBoxes': k_net.AnchorBoxes,
                                                                         'L2Normalization': k_net.L2Normalization,
                                                                         'DecodeDetections': k_net.DecodeDetections,
                                                                         'compute_loss': ssd_loss.compute_loss})
val_dataset = k_data.DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset.parse_xml(images_dirs=[os.path.join("/input/voc", "JPEGImages")],
                      image_set_filenames=[os.path.join("/input/voc", "MainSet", "det_test.txt")],
                      annotations_dirs=[os.path.join("/input/voc", "Annotations")],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=True,
                      ret=False)

evaluator = k_eval.Evaluator(model=model,
                      n_classes=20,
                      data_generator=val_dataset,
                      model_mode='inference')

results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=1,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.5,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)

mean_average_precision, average_precisions, precisions, recalls = results

for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
print()
print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))

# m = max((20 + 1) // 2, 2)
# n = 2
#
# fig, cells = plt.subplots(m, n, figsize=(n*8,m*8))
# for i in range(m):
#     for j in range(n):
#         if n*i+j+1 > 20: break
#         cells[i, j].plot(recalls[n*i+j+1], precisions[n*i+j+1], color='blue', linewidth=1.0)
#         cells[i, j].set_xlabel('recall', fontsize=14)
#         cells[i, j].set_ylabel('precision', fontsize=14)
#         cells[i, j].grid(True)
#         cells[i, j].set_xticks(np.linspace(0,1,11))
#         cells[i, j].set_yticks(np.linspace(0,1,11))
#         cells[i, j].set_title("{}, AP: {:.3f}".format(classes[n*i+j+1], average_precisions[n*i+j+1]), fontsize=16)

# evaluator.get_num_gt_per_class(ignore_neutral_boxes=True,verbose=False,ret=False)
# evaluator.match_predictions(ignore_neutral_boxes=True,matching_iou_threshold=0.5,border_pixels='include',
#                             sorting_algorithm='quicksort',verbose=True,ret=False)
# precisions, recalls = evaluator.compute_precision_recall(verbose=True, ret=True)
# average_precisions = evaluator.compute_average_precisions(mode='integrate',num_recall_points=11,verbose=True,ret=True)
# mean_average_precision = evaluator.compute_mean_average_precision(ret=True)
# for i in range(1, len(average_precisions)):
#     print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
# print()
# print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))