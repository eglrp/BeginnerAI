import torch
import PIL
import numpy as np
import scipy.misc as misc
import torchvision

import lib.fcn.dataset as fcn_data
import lib.fcn.net as fcn_net
import lib.fcn.loss as fcn_loss
import lib.ProgressBar as j_bar
import pydensecrf.densecrf as dcrf

FCNConfig = {
    "USE_GPU" : torch.cuda.is_available(),
    "CLASS_NUMS" : 21,
    "CLASSES" : ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'],
    "IMAGE_SIZE" : 512,
    "EPOCHS" : 120,
    "IMAGE_CHANNEL" : 3,
    "LEARNING_RATE" : 1.0e-10,
    "BATCH_SIZE" : 1,
    "MOMENTUM" : 0.99,
    "WEIGHT_DECAY" : 0.0005,
    "DATA_PATH" : "/input/VOC",
    "VGG16_MODEL_PATH" : "utils/vgg16_from_caffe.pth",
    "EPOCH" : 120
}

FROM_TRAIN_ITER = 1
data_set = fcn_data.VOCSegDataSet(FCNConfig["DATA_PATH"], is_transform=True, augmentations=None)
train_loader = torch.utils.data.DataLoader(data_set,batch_size=1, shuffle=True)

model = fcn_net.FCN8s(FCNConfig)
if FROM_TRAIN_ITER == 1:
    vgg = torchvision.models.vgg16(pretrained=False)
    state_dict = torch.load(FCNConfig["VGG16_MODEL_PATH"])
    vgg.load_state_dict(state_dict)
    model.init_vgg16_params(vgg)
else:
    model.load_state_dict(torch.load("outputs/FCN8s_%03d.pth" % (FROM_TRAIN_ITER - 1)))

if FCNConfig["USE_GPU"]:
    model = model.cuda()
LEARNING_RATE = FCNConfig["LEARNING_RATE"]
optim = torch.optim.SGD(model.parameters(),lr=FCNConfig["LEARNING_RATE"],momentum=FCNConfig["MOMENTUM"],
                        weight_decay=FCNConfig["WEIGHT_DECAY"])

bar = j_bar.ProgressBar(FCNConfig["EPOCH"], len(train_loader), "Loss : %.3f; Total Loss : %.3f")

for epoch in (FROM_TRAIN_ITER, FCNConfig["EPOCHS"] + 1):
    model.train()
    total_loss = 0

    for param_group in optim.param_groups:
        param_group['lr'] = LEARNING_RATE

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = torch.autograd.Variable(images.cuda() if FCNConfig["USE_GPU"] else images)
        targets = torch.autograd.Variable(targets.cuda() if FCNConfig["USE_GPU"] else targets)

        optim.zero_grad()
        score = model(images)
        loss = fcn_loss.cross_entropy2d(score, targets, size_average=False)
        total_loss += loss.item()

        bar.show(epoch, loss.item(), total_loss / (batch_idx + 1))

    torch.save(model.state_dict(), "outputs/FCN8s_%03d.pth" % epoch)

    model.eval()
    img = misc.imread("testImages/demo.jpg")
    resized_img = misc.imresize(
        img, (FCNConfig["IMAGE_SIZE"], FCNConfig["IMAGE_SIZE"]), interp="bicubic"
    )
    orig_size = img.shape[:-1]
    img = misc.imresize(img, (FCNConfig["IMAGE_SIZE"], FCNConfig["IMAGE_SIZE"]))
    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= data_set.mean
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()
    if FCNConfig["USE_GPU"]:
        img = img.cuda()

    pred = model(img)

    unary = pred.data.cpu().numpy()
    unary = np.squeeze(unary, 0)
    unary = -np.log(unary)
    unary = unary.transpose(2, 1, 0)
    w, h, c = unary.shape
    unary = unary.transpose(2, 0, 1).reshape(FCNConfig["CLASS_NUMS"], -1)
    unary = np.ascontiguousarray(unary)

    resized_img = np.ascontiguousarray(resized_img)

    d = dcrf.DenseCRF2D(w, h, FCNConfig["CLASS_NUMS"])
    d.setUnaryEnergy(unary)
    d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)

    q = d.inference(50)
    mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
    decoded_crf = data_set.decode_segmap(np.array(mask, dtype=np.uint8))

    misc.imsave("outputs/FCN8s_%03d.png" % epoch, decoded_crf)
