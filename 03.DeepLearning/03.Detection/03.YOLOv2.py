import torch as t
import lib.yolov2.utils as yolo_utils
import lib.yolov2.net as yolo_net
import lib.yolov2.dataset as yolo_data
import lib.ProgressBar as j_bar
import lib.yolov2.predict as yolo_predict
import time
import torchvision as tv

CONFIG = {
    "USE_CUDA" : t.cuda.is_available(),
    "NET_CONFIG" : "utils/yolov2_voc.cfg",
    "WEIGHT_FILE" : "utils/darknet19_448.conv.23",
    "TRAIN_LIST_FILE" : "utils/yolo2_train.txt",
    "EPOCHS" : 120,
    "CLASSES" : (  # always index 0
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')
}
FROM_TRAIN_ITER = 16

seed = int(time.time())
t.manual_seed(seed)

if CONFIG["USE_CUDA"]:
    t.cuda.manual_seed(seed)

net_options = yolo_utils.ConfigUtils.parse_cfg(CONFIG["NET_CONFIG"])[0]

BATCH_SIZE = int(net_options['batch'])
nsamples = yolo_utils.ConfigUtils.file_lines(CONFIG["TRAIN_LIST_FILE"])

model = yolo_net.Darknet(CONFIG["NET_CONFIG"])
region_loss = model.loss
model.load_weights(CONFIG["WEIGHT_FILE"])
model.print_network()

region_loss.seen = model.seen
processed_batches = model.seen / BATCH_SIZE

init_width = model.width
init_height = model.height
init_epoch = model.seen / nsamples
decay = float(net_options['decay'])
learning_rate = float(net_options['learning_rate'])
momentum = float(net_options['momentum'])
steps = [float(step) for step in net_options['steps'].split(',')]
scales = [float(scale) for scale in net_options['scales'].split(',')]

if CONFIG["USE_CUDA"]:
    model = model.cuda()
    region_loss.cuda()
params_dict = dict(model.named_parameters())
params = []
for key, value in params_dict.items():
    if key.find('.bn') >= 0 or key.find('.bias') >= 0:
        params += [{'params': [value], 'weight_decay': 0.0}]
    else:
        params += [{'params': [value], 'weight_decay': decay * BATCH_SIZE}]
optimizer = t.optim.SGD(model.parameters(), lr=learning_rate / BATCH_SIZE,
                      momentum=momentum, dampening=0, weight_decay=decay * BATCH_SIZE)

def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / BATCH_SIZE
    return lr

cur_model = model
train_loader = t.utils.data.DataLoader(
    yolo_data.listDataset(CONFIG["TRAIN_LIST_FILE"], shape=(init_width, init_height),
                        shuffle=True,
                        transform=tv.transforms.Compose([
                            tv.transforms.ToTensor(),
                            lambda x: 2 * (x - 0.5)
                        ]),
                        train=True,
                        seen=cur_model.seen,
                        batch_size=BATCH_SIZE),
    batch_size=BATCH_SIZE, shuffle=False)

predict = yolo_predict.YoloV2Predict(CONFIG["CLASSES"])

if FROM_TRAIN_ITER > 1:
    model.load_state_dict(t.load("outputs/YOLOV2_%03d.pth" % (FROM_TRAIN_ITER - 1)))

bar = j_bar.ProgressBar(CONFIG["EPOCHS"], len(train_loader), "Loss:%.3f;Avg Loss:%.3f")
for epoch in range(FROM_TRAIN_ITER, CONFIG["EPOCHS"] + 1):
    lr = adjust_learning_rate(optimizer, processed_batches)
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1

        data   = t.autograd.Variable(data.cuda() if CONFIG["USE_CUDA"] else data)
        target = t.autograd.Variable(target if CONFIG["USE_CUDA"] else target)
        optimizer.zero_grad()
        output = model(data)
        region_loss.seen = region_loss.seen + data.data.size(0)

        loss = region_loss(output, target) / BATCH_SIZE
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        bar.show(epoch, loss.item(), total_loss / (batch_idx + 1))

    t.save(model.state_dict(), "outputs/YOLOV2_%03d.pth" % epoch)
    model.eval()
    predict.predict(model, epoch, "testImages/demo.jpg")
