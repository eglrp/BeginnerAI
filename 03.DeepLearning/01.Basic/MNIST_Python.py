import lib.pytorch.cnn.python_layer as cnn_layer
import lib.pytorch.dataset.pytorch_dataset as j_data
import numpy as np
import lib.utils.ProgressBar as j_bar
import torch
import torchvision

CONFIG = {
    "EPOCHS" : 100,
    "BATCH_SIZE" : 64,
    "LEARNING_RATE" : 1e-4
}

dataset = j_data.MNISTDataSetForPytorch(radio=0.9, transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

conv1 = cnn_layer.Conv2D([CONFIG["BATCH_SIZE"], 28, 28, 1], 12, 5, 1)
relu1 = cnn_layer.Relu(conv1.output_shape)
pool1 = cnn_layer.MaxPooling(relu1.output_shape)
conv2 = cnn_layer.Conv2D(pool1.output_shape, 24, 3, 1)
relu2 = cnn_layer.Relu(conv2.output_shape)
pool2 = cnn_layer.MaxPooling(relu2.output_shape)
fc    = cnn_layer.FullyConnect(pool2.output_shape, 10)
sf    = cnn_layer.Softmax(fc.output_shape)

bar = j_bar.ProgressBar(CONFIG["EPOCHS"], len(train_loader), "train:%.3f,%.3f")

for epoch in range(1, CONFIG["EPOCHS"] + 1):
    batch_loss = 0
    batch_acc = 0
    val_acc = 0
    val_loss = 0

    # train
    train_acc = 0
    train_loss = 0
    for i, (train_image, train_label) in enumerate(train_loader):
        img = train_image.data.numpy().transpose((0, 2, 3, 1))
        label = train_label.data.numpy().squeeze()

        conv1_out = relu1.forward(conv1.forward(img))
        pool1_out = pool1.forward(conv1_out)
        conv2_out = relu2.forward(conv2.forward(pool1_out))
        pool2_out = pool2.forward(conv2_out)
        fc_out = fc.forward(pool2_out)
        batch_loss += sf.cal_loss(fc_out, np.array(label))
        train_loss = sf.cal_loss(fc_out, np.array(label))
        train_acc = 0
        for j in range(train_image.shape[0]):
            if np.argmax(sf.softmax[j]) == label[j]:
                train_acc += 1

        sf.gradient()
        conv1.gradient(relu1.gradient(pool1.gradient(
            conv2.gradient(relu2.gradient(pool2.gradient(
                fc.gradient(sf.eta)))))))

        if i % 1 == 0:
            fc.backward(alpha=CONFIG["LEARNING_RATE"], weight_decay=0.0004)
            conv2.backward(alpha=CONFIG["LEARNING_RATE"], weight_decay=0.0004)
            conv1.backward(alpha=CONFIG["LEARNING_RATE"], weight_decay=0.0004)
            batch_loss = 0
            batch_acc = 0

        bar.show(epoch, train_loss / (train_image.shape[0]), train_acc / (train_image.shape[0]))

