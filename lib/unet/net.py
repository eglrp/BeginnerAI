import torch

class UNet(torch.nn.Module):
    def __init__(self, CFG):
        super(UNet, self).__init__()
        self.CFG = CFG
        self.is_deconv = True
        self.in_channel = CFG["IMAGE_CHANNEL"]
        self.is_batchnorm = True
        self.feature_scale = CFG["FEATURE_SCALE"]

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1    = UNetConv2(self.in_channel, filters[0], self.is_batchnorm)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv2    = UNetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv3    = UNetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2)

        self.conv4    = UNetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=2)

        self.center = UNetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UNetUp(filters[4], filters[3])
        self.up_concat3 = UNetUp(filters[3], filters[2])
        self.up_concat2 = UNetUp(filters[2], filters[1])
        self.up_concat1 = UNetUp(filters[1], filters[0])

        # final conv (without any concat)
        self.final = torch.nn.Conv2d(filters[0], CFG["CLASS_NUMS"], 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final

class UNetConv2(torch.nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(UNetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_size, out_size, 3, 1, 1),
                torch.nn.BatchNorm2d(out_size),
                torch.nn.ReLU())
            self.conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(out_size, out_size, 3, 1, 1),
                torch.nn.BatchNorm2d(out_size),
                torch.nn.ReLU())
        else:
            self.conv1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_size, out_size, 3, 1, 1),
                torch.nn.ReLU())
            self.conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(out_size, out_size, 3, 1, 1),
                torch.nn.ReLU())
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class UNetUp(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.conv = UNetConv2(in_size, out_size, False)
        self.up = torch.nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = torch.nn.functional.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))