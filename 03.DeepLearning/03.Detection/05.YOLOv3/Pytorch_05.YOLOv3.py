import lib.yolov3.utils as yolo_utils
import lib.yolov3.net as yolo_net
import torch
import time
import lib.ProgressBar as j_bar
import lib.yolov3.dataset as yolo_data
import torchvision
import lib.yolov3.predict as yolo_predict
import cv2
import lib.utils.logger as logger
import os
import tqdm

CONFIG = {
    "DATA_CONFIG_FILE" : "utils/voc.data",
    "CONFIG_FILE" : "utils/yolo_v3.cfg",
    "IMAGE_SIZE" : 416,
    "EPOCHS" : 200,
    "WEIGHTS_FILE" : "utils/yolov3.weights",
    "USE_GPU" : torch.cuda.is_available(),
    "CLASSES" : (  # always index 0
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')
}
FROM_TRAIN_ITER = 1
data_options  = yolo_utils.LoadUtils.read_data_cfg(CONFIG["DATA_CONFIG_FILE"])
net_options   = yolo_utils.LoadUtils.parse_cfg(CONFIG["CONFIG_FILE"])[0]
weightfile = CONFIG["WEIGHTS_FILE"]

trainlist     = data_options["train"]
batch_size    = int(net_options['batch'])
max_batches   = int(net_options['max_batches'])
learning_rate = float(net_options['learning_rate'])
momentum      = float(net_options['momentum'])
decay         = float(net_options['decay'])
steps         = [float(step) for step in net_options['steps'].split(',')]
scales        = [float(scale) for scale in net_options['scales'].split(',')]

try:
    max_epochs = int(net_options['max_epochs'])
except KeyError:
    max_epochs = CONFIG["EPOCHS"]

seed = int(time.time())
torch.manual_seed(seed)
if CONFIG["USE_GPU"]:
    torch.cuda.manual_seed(seed)
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

model = yolo_net.Darknet(CONFIG["CONFIG_FILE"], use_cuda=CONFIG["USE_GPU"])
# model.load_weights(weightfile)
nsamples = yolo_utils.LoadUtils.file_lines(trainlist)
model.seen = 0
init_epoch = 0
loss_layers = model.loss_layers

if CONFIG["USE_GPU"]:
    model = model.cuda()

init_width = model.width
init_height = model.height

optimizer = torch.optim.SGD(model.parameters(),
                            lr=learning_rate, momentum=momentum,
                            dampening=0, weight_decay=decay*batch_size)
train_loader = torch.utils.data.DataLoader(
    yolo_data.listDataset(trainlist, shape=(init_width, init_height),
                          shuffle=True,
                          transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor(),
                          ]),
                          train=True,
                          seen=model.seen,
                          batch_size=batch_size),
    batch_size=batch_size, shuffle=False)

def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            print("learning rate down")
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr
processed_batches = 0
if FROM_TRAIN_ITER > 1:
    model.load_state_dict(torch.load("outputs/YOLOV3_%03d.pth" % (FROM_TRAIN_ITER - 1)))
log = logger.Logger("logs/")
predict = yolo_predict.YoloV3Predict(CONFIG["CLASSES"])
LEARNING_RATE = learning_rate
bar = j_bar.ProgressBar(max_epochs, len(train_loader), "Loss:%.3f;Total Loss:%.3f")
for epoch in range(FROM_TRAIN_ITER, max_epochs + 1):
    model.train()
    total_loss = 0
    torch.cuda.empty_cache()
    # if epoch >= 1:
    #     LEARNING_RATE = 0.01
    # if epoch >= 30:
    #     LEARNING_RATE = 0.005
    # if epoch >= 60:
    #     LEARNING_RATE = 0.001
    # if epoch >= 90:
    #     LEARNING_RATE = 0.0005
    # if epoch >= 120:
    #     LEARNING_RATE = 0.00025
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = LEARNING_RATE/batch_size
    for batch_idx, (data, target) in enumerate(train_loader):
        processed_batches = model.seen//batch_size

        adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1
        images = torch.autograd.Variable(data.cuda() if CONFIG["USE_GPU"] else data)
        target = torch.autograd.variable(target.cuda() if CONFIG["USE_GPU"] else target)

        optimizer.zero_grad()

        output = model(data)

        org_loss = []
        for i, l in enumerate(loss_layers):
            l.seen = l.seen + data.data.size(0)
            ol=l(output[i]['x'], target)
            org_loss.append(ol)

        loss = sum(org_loss) / batch_size
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        bar.show(epoch, loss.item(), total_loss / (batch_idx + 1))

    torch.save(model.state_dict(), "outputs/YOLOV3_%03d.pth" % epoch)

    model.eval()

    path = os.path.join("testImages")
    listfile = os.listdir(path)
    for file in tqdm.tqdm(listfile):
        if file.endswith("jpg"):
            filename = file.split(".")[0]
            image = predict.predict(model, epoch, os.path.join(path,"%s.jpg" % filename), filename, targetPath="results/")

    # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    info = {'loss': total_loss / (batch_idx+1)}

    for tag, value in info.items():
        log.scalar_summary(tag, value, epoch)

    # imageInfo = {'images': image}
    # for tag, value in imageInfo.items():
    #     log.image_summary(tag, value, epoch)