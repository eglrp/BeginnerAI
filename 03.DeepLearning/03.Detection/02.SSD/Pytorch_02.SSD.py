import torch as t
import cv2
import lib.pytorch.utils.logger as logger
import lib.pytorch.ssd.augmentations as ssd_aug
import lib.pytorch.ssd.dataset as ssd_data
import lib.pytorch.ssd.loss as ssd_loss
import lib.pytorch.ssd.net as ssd_net
import lib.pytorch.ssd.predict as ssd_predict
import lib.pytorch.ssd.utils as ssd_utils

import lib.utils.ProgressBar as j_bar

CONFIG = {
    "USE_GPU" : t.cuda.is_available(),
    "DATA_PATH" : "/input/VOC/",
    "IMAGE_SIZE" : 300,
    "IMAGE_CHANNEL" : 3,
    "EPOCH" : 300,
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
    "LOG_DIR" : "logs/",
    "CLASSES" : (  # always index 0
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')
}

Color = [[0, 0, 0],
         [128, 0, 0],
         [0, 128, 0],
         [128, 128, 0],
         [0, 0, 128],
         [128, 0, 128],
         [0, 128, 128],
         [128, 128, 128],
         [64, 0, 0],
         [192, 0, 0],
         [64, 128, 0],
         [192, 128, 0],
         [64, 0, 128],
         [192, 0, 128],
         [64, 128, 128],
         [192, 128, 128],
         [0, 64, 0],
         [128, 64, 0],
         [0, 192, 0],
         [128, 192, 0],
         [0, 64, 128]]

FROM_TRAIN_ITER = 1
if CONFIG["USE_GPU"]:
    t.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    t.set_default_tensor_type('torch.FloatTensor')

dataset = ssd_data.VOCDetection(transform=ssd_aug.SSDAugmentation(CONFIG["IMAGE_SIZE"],CONFIG["MEANS"]),
                              target_transform=ssd_data.VOCAnnotationTransform(CONFIG["CLASSES"]))

net = ssd_net.build_ssd("train", CONFIG)
vgg_weights = t.load("utils/SSD300_PreTrained_VGG.pth")

net.vgg.load_state_dict(vgg_weights)

# 使用xavier方法来初始化vgg后面的新增层、loc用于回归层、conf用于分类层  的权重
net.extras.apply(ssd_utils.weights_init)
net.loc.apply(ssd_utils.weights_init)
net.conf.apply(ssd_utils.weights_init)

LEARNING_RATE = CONFIG["LEARNING_RATE"]
optimizer = t.optim.SGD(net.parameters(), lr=CONFIG["LEARNING_RATE"], momentum=CONFIG["MOMENTUM"],
                        weight_decay=CONFIG["WEIGHT_DECAY"])
# SSD的损失函数
criterion = ssd_loss.MultiBoxLoss(CONFIG, CONFIG['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                False, CONFIG["USE_GPU"])

train_loader = t.utils.data.DataLoader(dataset, CONFIG["BATCH_SIZE"],
                                       shuffle=True, collate_fn=ssd_utils.detection_collate)

if FROM_TRAIN_ITER > 1:
    net.load_state_dict(t.load("outputs/SSD_%03d.pth" % (FROM_TRAIN_ITER - 1)))

index = 0
step_index = 0
# predict = j_m_ssd.SSDPredict(CONFIG["CLASSES"])
bar = j_bar.ProgressBar(CONFIG["EPOCH"], len(train_loader), "Loss : %.3f; Total Loss : %.3f")

predict = ssd_predict.SSDPredict(CONFIG["CLASSES"])
net.train()
log = logger.Logger(CONFIG["LOG_DIR"])
for epoch in range(FROM_TRAIN_ITER, CONFIG["EPOCH"] + 1):
    total_loss = 0.
    t.cuda.empty_cache()
    for i, (images, targets) in enumerate(train_loader):
        index += i
        if epoch >= 30:
            LEARNING_RATE = 0.0005
        if epoch >= 50:
            LEARNING_RATE = 0.00025
        if epoch >= 80:
            LEARNING_RATE = 0.00001
        if epoch >= 100:
            LEARNING_RATE = 0.000005
        for param_group in optimizer.param_groups:
            param_group['lr'] = LEARNING_RATE

        images = t.autograd.Variable(images.cuda() if CONFIG["USE_GPU"] else images)
        targets = [t.autograd.Variable(ann.cuda() if CONFIG["USE_GPU"] else ann) for ann in targets]

        out = net(images)

        optimizer.zero_grad()

        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        bar.show(epoch, loss.item(), total_loss / (i+1))

    t.save(net.state_dict(),"outputs/SSD_%03d.pth" % epoch)

    test_net = ssd_net.build_ssd('test', CONFIG)  # 初始化 SSD300，类别为21（20类别+1背景）
    test_net.load_state_dict(t.load("outputs/SSD_%03d.pth" % epoch, map_location=lambda storage, loc: storage))

    image = predict.predict(test_net, epoch, "testImages/03.jpg", "demo")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    info = {'loss': total_loss / (i+1)}

    for tag, value in info.items():
        log.scalar_summary(tag, value, epoch)

    imageInfo = {'images': image}
    for tag, value in imageInfo.items():
        log.image_summary(tag, value, epoch)