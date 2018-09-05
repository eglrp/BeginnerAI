'''
https://github.com/kuangliu/pytorch-retinanet
'''
import torch
import torchvision
import lib.retinanet.dataset as rn_data
import os
import lib.retinanet.net as rn_net
import lib.retinanet.loss as rn_loss
import lib.ProgressBar as j_bar
import lib.retinanet.predict as rn_predict
CONFIG = {
    "LEARNING_RATE" : 1e-3,
    "DATA_FOLDER" : os.path.join("/input", "VOC2012", "JPEGImages"),
    "IMAGE_SIZE" : 600,
    "BATCH_SIZE" : 16,
    "USE_GPU" : torch.cuda.is_available(),
    "EPOCHS" : 120,
    "CLASSES" : (  # always index 0
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')
}
FROM_TRAIN_ITER = 1
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

trainset = rn_data.ListDataset(root=CONFIG["DATA_FOLDER"],
                       list_file='utils/retinanet_voc_2012.txt', train=True, transform=transform,
                       input_size=CONFIG["IMAGE_SIZE"])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True,
                                          collate_fn=trainset.collate_fn)

net = rn_net.RetinaNet()
if FROM_TRAIN_ITER == 1:
    net.load_state_dict(torch.load('utils/retinanet_pre.pth'))
if CONFIG["USE_GPU"]:
    net.cuda()

criterion = rn_loss.FocalLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=CONFIG["LEARNING_RATE"], momentum=0.9, weight_decay=1e-4)
if FROM_TRAIN_ITER > 1:
    net.load_state_dict(torch.load("outputs/RetinaNet_%03d.pth" % (FROM_TRAIN_ITER - 1)))

predict = rn_predict.RetinaPredict(CONFIG["CLASSES"])
bar = j_bar.ProgressBar(CONFIG["EPOCHS"], len(trainloader), "Loss:%.3f;Total Loss:%.3f")
for epoch in range(FROM_TRAIN_ITER, CONFIG["EPOCHS"] + 1):
    net.train()
    total_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs      = torch.autograd.Variable(inputs.cuda() if torch.cuda.is_available() else inputs)
        loc_targets = torch.autograd.Variable(loc_targets.cuda() if torch.cuda.is_available() else loc_targets)
        cls_targets = torch.autograd.Variable(cls_targets.cuda() if torch.cuda.is_available() else cls_targets)

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        bar.show(epoch, loss.item(), total_loss / (batch_idx + 1))

    torch.save(net.state_dict(), "outputs/RetinaNet_%03d.pth" % epoch)

    net.eval()
    predict.predict(net, epoch, "testImages/demo.jpg")