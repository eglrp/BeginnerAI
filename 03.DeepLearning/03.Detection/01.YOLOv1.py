import torch as t
import torchvision as tv

import lib.yolov1.model as yolo_model
import lib.yolov1.loss as yolo_loss
import lib.yolov1.dataset as yolo_data
import lib.yolov1.predict as yolo_predict

import lib.ProgressBar as j_bar

CONFIG = {
    "USE_GPU" : t.cuda.is_available(),
    "EPOCHS" : 120,
    "IMAGE_SIZE" : 448,
    "IMAGE_CHANNEL" : 3,
    "ALPHA" : 0.1,
    "BATCH_SIZE" : 32,
    "DATA_PATH" : "/input/VOC2012/JPEGImages",
    "CELL_NUMS" : 7,
    "CLASS_NUMS" : 20,
    "BOXES_EACH_CELL" : 2,
    "LEARNING_RATE" : 0.001,
    "L_COORD" : 5,
    "L_NOOBJ" : 0.5,
    "CLASSES" : ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor']
}

FROM_TRAIN_ITER = 1

Net = yolo_model.YoLoV1Net()

criterion = yolo_loss.YoLoV1Loss(7, 2, 5, 0.5)

if CONFIG["USE_GPU"]:
    Net.cuda()
    criterion.cuda()

train_dataset = yolo_data.yoloDataset(root=CONFIG["DATA_PATH"],
                                  list_file="utils/voc2012train.txt", train=True,
                                  transform=[tv.transforms.ToTensor()])

train_loader = t.utils.data.DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

# 优化器
optimizer = t.optim.SGD(Net.parameters(), lr=CONFIG["LEARNING_RATE"],
                        momentum=0.95, weight_decay=5e-4)
bar = j_bar.ProgressBar(CONFIG["EPOCHS"], len(train_loader), "Loss : %.3f; Total Loss : %.3f")

if FROM_TRAIN_ITER > 1:
    Net.load_state_dict(t.load("outputs/yolov1_%03d.pth" % (FROM_TRAIN_ITER - 1)))

predict = yolo_predict.YoLoPredict(CONFIG["CLASSES"])
for epoch in range(FROM_TRAIN_ITER, CONFIG["EPOCHS"] + 1):
    Net.train()
    if epoch == 1:
        LEARNING_RATE = 0.0005
    if epoch == 2:
        LEARNING_RATE = 0.00075
    if epoch == 3:
        LEARNING_RATE = 0.001
    if epoch == 80:
        LEARNING_RATE = 0.0001
    if epoch >= 100:
        LEARNING_RATE = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = LEARNING_RATE

    total_loss = 0.
    # 开始训练
    for i, (images, target) in enumerate(train_loader):
        images = t.autograd.Variable(images.cuda() if CONFIG["USE_GPU"] else images)
        target = t.autograd.Variable(target.cuda() if CONFIG["USE_GPU"] else target)
        if CONFIG["USE_GPU"]:
            images, target = images.cuda(), target.cuda()
        pred = Net(images)
        loss = criterion(pred, target)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bar.show(epoch, loss.item(), total_loss / (i+1))

    t.save(Net.state_dict(),"outputs/yolov1_%03d.pth" % epoch)

    Net.eval()
    predict.predict(Net, epoch, "testImages/demo.jpg")