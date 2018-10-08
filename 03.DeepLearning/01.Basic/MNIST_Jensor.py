import lib.jensor.jensor as jen
import lib.jensor.layers as lay

import lib.dataset.pytorch_dataset as j_data
import numpy as np
import lib.ProgressBar as j_bar
import torch
import torchvision

CONFIG = {
    "EPOCHS" : 100,
    "BATCH_SIZE" : 64,
    "LEARNING_RATE" : 1e-4
}

dataset = j_data.MNISTDataSetForPytorch(radio=0.9, transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

def inference(x, output_num):
    conv1_out = lay.Conv2D((5, 5, 1, 12), input_variable=x, name='conv1', padding='VALID').output_variables
    relu1_out = lay.Relu(input_variable=conv1_out, name='relu1').output_variables
    pool1_out = lay.MaxPooling(ksize=2, input_variable=relu1_out, name='pool1').output_variables

    conv2_out = lay.Conv2D((3, 3, 12, 24), input_variable=pool1_out, name='conv2').output_variables
    relu2_out = lay.Relu(input_variable=conv2_out, name='relu2').output_variables
    pool2_out = lay.MaxPooling(ksize=2, input_variable=relu2_out, name='pool2').output_variables

    fc_out    = lay.FullyConnect(output_num=output_num, input_variable=pool2_out, name='fc').output_variables
    return fc_out

for k in jen.GLOBAL_VARIABLE_SCOPE:
    s = jen.GLOBAL_VARIABLE_SCOPE[k]
    if isinstance(s, jen.Variable) and s.learnable:
        s.set_method_adam()

img_placeholder   = jen.Variable((CONFIG["BATCH_SIZE"], 28, 28, 1), 'input')
label_placeholder = jen.Variable([CONFIG["BATCH_SIZE"], 1], 'label')

# set train_op
prediction = inference(img_placeholder, 10)
sf = lay.SoftmaxLoss(prediction, label_placeholder, 'sf')

bar = j_bar.ProgressBar(CONFIG["EPOCHS"], len(train_loader), "train:%.3f,%.3f")
global_step = 0
for epoch in range(1, CONFIG["EPOCHS"] + 1):
    batch_loss = 0
    batch_acc = 0
    val_acc = 0
    val_loss = 0

    # train
    train_acc = 0
    train_loss = 0
    for i, (train_image, train_label) in enumerate(train_loader):
        learning_rate = 5e-4 * pow(0.1 , float(epoch/10))

        img = train_image.data.numpy().transpose((0, 2, 3, 1))
        label = train_label.data.numpy().squeeze()

        img_placeholder.data = img
        label_placeholder.data = label

        # forward
        _loss = sf.loss.eval()
        _prediction = sf.prediction.eval()
        train_loss = _loss
        train_acc = 0

        for j in range(train_image.shape[0]):
            if np.argmax(sf.softmax[j]) == label[j]:
                train_acc += 1

        img_placeholder.diff_eval()

        for k in jen.GLOBAL_VARIABLE_SCOPE:
            s = jen.GLOBAL_VARIABLE_SCOPE[k]
            if isinstance(s, jen.Variable) and s.learnable:
                s.apply_gradient(learning_rate=learning_rate, decay_rate=0.0004, batch_size=CONFIG["BATCH_SIZE"])
            if isinstance(s, jen.Variable):
                s.diff = np.zeros(s.shape)
            global_step += 1

        bar.show(epoch, train_loss / (train_image.shape[0]), train_acc / (train_image.shape[0]))

