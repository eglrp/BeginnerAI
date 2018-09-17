import lib.ssd.predict as ssd_predict
import lib.ssd.net as ssd_net
import torch as t
import os
import tqdm

PHRASE = "Predict" # "Test"

CONFIG = {
    "USE_GPU" : t.cuda.is_available(),
    "DATA_PATH" : "/input",
    "IMAGE_SIZE" : 300,
    "IMAGE_CHANNEL" : 3,
    "EPOCH" : 120,
    "LEARNING_RATE" : 0.001,
    "MOMENTUM" : 0.9,
    "WEIGHT_DECAY" : 5e-4,
    "BATCH_SIZE" : 32,
    "GAMMA" : 0.1,
    "MEANS" : (104, 117, 123),
    'num_classes': 21,  # 分类类别20+背景1
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,  # 迭代次数
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,  # 当前SSD300只支持大小300×300的数据集训练
    'steps': [8, 16, 32, 64, 100, 300],  # 感受野，相对于原图缩小的倍数
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],  # 方差
    'clip': True,
    'name': 'VOC',
    "CLASSES" : (  # always index 0
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')
}

model = ssd_net.build_ssd("test", CONFIG)
model.load_state_dict(t.load("results/SSD01.pth", map_location={'cuda:0': 'cpu'}))

if t.cuda.is_available():
    model = model.cuda()

predict = ssd_predict.SSDPredict(CONFIG["CLASSES"])
model.eval()

if PHRASE == "Predict":
    path = os.path.join("../testImages")
    listfile = os.listdir(path)
    for file in tqdm.tqdm(listfile):
        if file.endswith("jpg"):
            filename = file.split(".")[0]
            predict.predict(model, 0, os.path.join(path,"%s.jpg" % filename), filename, targetPath="results/")
else:
    pass