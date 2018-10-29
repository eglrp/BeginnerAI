import torch
import PIL
import numpy as np
import scipy.misc as misc
import torchvision

import lib.pytorch.unet.dataset as unet_data
import lib.pytorch.unet.net as unet_net
import lib.pytorch.unet.loss as unet_loss
import lib.utils.ProgressBar as j_bar
import pydensecrf.densecrf as dcrf

CONFIG = {
    "USE_GPU" : torch.cuda.is_available(),
    "CLASS_NUMS" : 21,
    "CLASSES" : ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'],
    "IMAGE_SIZE" : 256,
    "EPOCHS" : 120,
    "IMAGE_CHANNEL" : 3,
    "LEARNING_RATE" : 1.0e-10,
    "BATCH_SIZE" : 1,
    "MOMENTUM" : 0.99,
    "WEIGHT_DECAY" : 0.0005,
    "DATA_PATH" : "/input/VOC2012",
    "EPOCH" : 120,
    "FEATURE_SCALE" : 4
}

FROM_TRAIN_ITER = 1
data_set = unet_data.VOCSegDataSet(CONFIG["DATA_PATH"], is_transform=True, augmentations=None,
                                   img_size=(CONFIG["IMAGE_SIZE"],CONFIG["IMAGE_SIZE"]))
train_loader = torch.utils.data.DataLoader(data_set,batch_size=1, shuffle=True)

model = unet_net.UNet(CONFIG)
if FROM_TRAIN_ITER > 1:
    model.load_state_dict(torch.load("outputs/UNet_%03d.pth" % (FROM_TRAIN_ITER - 1)))

if CONFIG["USE_GPU"]:
    model = model.cuda()
LEARNING_RATE = CONFIG["LEARNING_RATE"]
optim = torch.optim.SGD(model.parameters(),lr=CONFIG["LEARNING_RATE"],momentum=CONFIG["MOMENTUM"],
                        weight_decay=CONFIG["WEIGHT_DECAY"])

bar = j_bar.ProgressBar(CONFIG["EPOCH"], len(train_loader), "Loss : %.3f; Total Loss : %.3f")

for epoch in (FROM_TRAIN_ITER, CONFIG["EPOCHS"] + 1):
    model.train()
    total_loss = 0

    for param_group in optim.param_groups:
        param_group['lr'] = LEARNING_RATE

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = torch.autograd.Variable(images.cuda() if CONFIG["USE_GPU"] else images)
        targets = torch.autograd.Variable(targets.cuda() if CONFIG["USE_GPU"] else targets)

        optim.zero_grad()
        score = model(images)
        loss = unet_loss.cross_entropy2d(score, targets, size_average=False)
        total_loss += loss.item()

        bar.show(epoch, loss.item(), total_loss / (batch_idx + 1))

    torch.save(model.state_dict(), "outputs/UNet_%03d.pth" % epoch)
    model.eval()
    img = misc.imread("testImages/demo.jpg")
    resized_img = misc.imresize(
        img, (CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"]), interp="bicubic"
    )
    orig_size = img.shape[:-1]
    img = misc.imresize(img, (CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"]))
    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= data_set.mean
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()
    if CONFIG["USE_GPU"]:
        img = img.cuda()

    pred = model(img)

    unary = pred.data.cpu().numpy()
    unary = np.squeeze(unary, 0)
    unary = -np.log(unary)
    unary = unary.transpose(2, 1, 0)
    w, h, c = unary.shape
    unary = unary.transpose(2, 0, 1).reshape(CONFIG["CLASS_NUMS"], -1)
    unary = np.ascontiguousarray(unary)

    resized_img = np.ascontiguousarray(resized_img)

    d = dcrf.DenseCRF2D(w, h, CONFIG["CLASS_NUMS"])
    d.setUnaryEnergy(unary)
    d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)

    q = d.inference(50)
    mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
    decoded_crf = data_set.decode_segmap(np.array(mask, dtype=np.uint8))

    misc.imsave("outputs/UNet_%03d.png" % epoch, decoded_crf)
