import torch as t
import torchvision as tv

import torch.utils.data as t_data
import torch.autograd as t_auto
import torch.optim as t_optim

import lib.utils.ProgressBar as j_bar

CONFIG = {
    "DATA_PATH" : "/input/JData/",
    "EPOCH" : 100,
    "GPU_NUMS" : 1,
    "BATCH_SIZE" : 64,
    "NOISE_DIM" : 100,
    "IMAGE_CHANNEL": 3,
    "IMAGE_SIZE" : 64,
    "LEARNING_RATE" : 2e-4
}

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, t.nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, t.nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, t.nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


class Generator(t.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = t.nn.Sequential(
            t.nn.Linear(CONFIG["NOISE_DIM"] + 10, 1024),
            t.nn.BatchNorm1d(1024),
            t.nn.ReLU()
        )

        self.fc2 = t.nn.Sequential(
            t.nn.Linear(1024, 128 * 8 * 8),
            t.nn.BatchNorm1d(128 * 8 * 8),
            t.nn.ReLU()
        )

        self.deconv1 = t.nn.Sequential(
            t.nn.ConvTranspose2d(128, 64, 4, 2, 1),
            t.nn.BatchNorm2d(64),
            t.nn.ReLU()
        )

        self.deconv2 = t.nn.Sequential(
            t.nn.ConvTranspose2d(64, 32, 4, 2, 1),
            t.nn.BatchNorm2d(32),
            t.nn.ReLU()
        )

        self.deconv3 = t.nn.Sequential(
            t.nn.ConvTranspose2d(32, CONFIG["IMAGE_CHANNEL"], 4, 2, 1),
            t.nn.Sigmoid()
        )

        initialize_weights(self)

    def forward(self, input, label):
        x = t.cat([input, label], 1)
        x = x.view(-1,CONFIG["NOISE_DIM"] + 10)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 128, 8, 8)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)

        return x

class Discriminator(t.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(CONFIG["IMAGE_CHANNEL"] + 10, 64, 4, 2, 1),
            t.nn.LeakyReLU(0.2)
        )

        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(64, 128, 4, 2, 1),
            t.nn.BatchNorm2d(128),
            t.nn.LeakyReLU(0.2)
        )

        self.conv3 = t.nn.Sequential(
            t.nn.Conv2d(128, 256, 4, 2, 1),
            t.nn.BatchNorm2d(256),
            t.nn.LeakyReLU(0.2)
        )

        self.fc1 = t.nn.Sequential(
            t.nn.Linear(256 * 8 * 8, 1024),
            t.nn.BatchNorm1d(1024),
            t.nn.LeakyReLU(0.2)
        )

        self.fc2 = t.nn.Sequential(
            t.nn.Linear(1024, 1),
            t.nn.Sigmoid()
        )
        initialize_weights(self)

    def forward(self, input, label):
        output = t.cat([input, label], 1)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(-1, 256 * 8 * 8)
        output = self.fc1(output)
        output = self.fc2(output)

        return output

NetD = Discriminator()
NetG = Generator()
MSE_LOSS = t.nn.MSELoss()
if CONFIG["GPU_NUMS"] > 0:
    NetG = NetG.cuda()
    NetD = NetD.cuda()
    MSE_LOSS = MSE_LOSS.cuda()

optimizerD = t_optim.Adam(NetD.parameters(), lr=CONFIG["LEARNING_RATE"], betas=(0.5, 0.999))
optimizerG = t_optim.Adam(NetG.parameters(), lr=CONFIG["LEARNING_RATE"], betas=(0.5, 0.999))

dataset = tv.datasets.ImageFolder(root=CONFIG["DATA_PATH"], transform=tv.transforms.Compose([
    tv.transforms.Resize((CONFIG["IMAGE_SIZE"],CONFIG["IMAGE_SIZE"])),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.5] * 3, [0.5] * 3)
]))
train_loader = t_data.DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

def one_hot(target):
    y = t.zeros(target.size()[0], 10)

    for i in range(target.size()[0]):
        y[i, target[i]] = 1

    return y

Predict_Noise_var = t_auto.Variable(t.randn(100, CONFIG["NOISE_DIM"],1,1).cuda() if CONFIG["GPU_NUMS"] > 0 else t.randn(100, CONFIG["NOISE_DIM"],1,1))
temp_z_ = t.randn(10, 100)
fixed_z_ = temp_z_
Predict_y = t.zeros(10, 1)

fixed_z_ = t.cat([fixed_z_, temp_z_], 0)
temp     = t.ones(10, 1) + 0
Predict_y= t.cat([Predict_y, temp], 0)
temp     = t.ones(10, 1) + 1
Predict_y= t.cat([Predict_y, temp], 0)

Predict_y= t.cat([Predict_y, t.zeros(10, 1)], 0)
fixed_z_ = t.cat([fixed_z_, temp_z_], 0)
temp     = t.ones(10, 1) + 0
Predict_y= t.cat([Predict_y, temp], 0)
temp     = t.ones(10, 1) + 1
Predict_y= t.cat([Predict_y, temp], 0)

Predict_y= t.cat([Predict_y, t.zeros(10, 1)], 0)
fixed_z_ = t.cat([fixed_z_, temp_z_], 0)
temp     = t.ones(10, 1) + 0
Predict_y= t.cat([Predict_y, temp], 0)
temp     = t.ones(10, 1) + 1
Predict_y= t.cat([Predict_y, temp], 0)

Predict_y= t.cat([Predict_y, t.zeros(10, 1)], 0)

Predict_y= one_hot(Predict_y.long())
Predict_y = Predict_y.view(-1, 10, 1,1)
Predict_y= t_auto.Variable(Predict_y.cuda() if CONFIG["GPU_NUMS"] > 0 else Predict_y)

bar = j_bar.ProgressBar(CONFIG["EPOCH"], len(train_loader), "D Loss:%.3f;G Loss:%.3f")

fill = t.zeros([10, 10, CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"]])
for i in range(10):
    fill[i, i, :, :] = 1
onehot = t.zeros(10, 10)
onehot = onehot.scatter_(1, t.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)

NetD.train()
for epoch in range(1, CONFIG["EPOCH"] + 1):
    NetG.train()
    for index, (image, label) in enumerate(train_loader):
        mini_batch = label.shape[0]

        label_true_var  = t_auto.Variable(t.ones(mini_batch).cuda() if CONFIG["GPU_NUMS"] > 0 else t.ones(mini_batch))
        label_false_var = t_auto.Variable(t.zeros(mini_batch).cuda() if CONFIG["GPU_NUMS"] > 0 else t.zeros(mini_batch))

        NetD.zero_grad()
        label_real = label.squeeze().type(t.LongTensor)
        label_real = fill[label_real]

        image_var = t_auto.Variable(image.cuda() if CONFIG["GPU_NUMS"] > 0 else image)
        label_var = t_auto.Variable(label_real.cuda() if CONFIG["GPU_NUMS"] > 0 else label_real)

        d_result = NetD(image_var, label_var)
        d_result = d_result.squeeze()
        D_LOSS_REAL = MSE_LOSS(d_result, label_true_var)

        img_fake = t.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        label_fake = (t.rand(mini_batch, 1) * 10).type(t.LongTensor).squeeze()
        img_fake_var = t_auto.Variable(img_fake.cuda() if CONFIG["GPU_NUMS"] > 0 else img_fake)
        label_fake_G_var = t_auto.Variable(onehot[label_fake].cuda() if CONFIG["GPU_NUMS"] > 0 else onehot[label_fake])
        label_fake_D_var = t_auto.Variable(fill[label_fake].cuda() if CONFIG["GPU_NUMS"] > 0 else fill[label_fake])

        g_result = NetG(img_fake_var, label_fake_G_var)
        d_result = NetD(g_result, label_fake_D_var)
        d_result = d_result.squeeze()
        D_LOSS_FAKE = MSE_LOSS(d_result, label_false_var)

        D_train_loss = D_LOSS_REAL + D_LOSS_FAKE
        D_train_loss.backward()
        optimizerD.step()

        NetG.zero_grad()
        img_fake = t.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        label_fake = (t.rand(mini_batch, 1) * 10).type(t.LongTensor).squeeze()
        img_fake_var = t_auto.Variable(img_fake.cuda() if CONFIG["GPU_NUMS"] > 0 else img_fake)
        label_fake_G_var = t_auto.Variable(onehot[label_fake].cuda() if CONFIG["GPU_NUMS"] > 0 else onehot[label_fake])
        label_fake_D_var = t_auto.Variable(fill[label_fake].cuda() if CONFIG["GPU_NUMS"] > 0 else fill[label_fake])
        g_result = NetG(img_fake_var, label_fake_G_var)
        d_result = NetD(g_result, label_fake_D_var)
        d_result = d_result.squeeze()
        G_train_loss= MSE_LOSS(d_result, label_true_var)
        G_train_loss.backward()
        optimizerG.step()

        bar.show(epoch, D_train_loss.item(), G_train_loss.item())

    fake_u=NetG(Predict_Noise_var, Predict_y)
    tv.utils.save_image(fake_u.data,'outputs/CFA_%03d.png' % epoch,nrow=10)

