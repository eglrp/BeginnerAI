import torch
import numpy as np

class FCNs(torch.nn.Module):
    def __init__(self, cfg):
        super(FCNs, self).__init__()
        self.CFG = cfg
        self.n_classes = self.CFG["CLASS_NUMS"]
        self.Conv_Block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.CFG["IMAGE_CHANNEL"], out_channels=64, kernel_size=3, padding=100),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        self.Conv_Block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        self.Conv_Block3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        self.Conv_Block4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        self.Conv_Block5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout2d(),
            torch.nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout2d(),
            torch.nn.Conv2d(in_channels=4096, out_channels=self.CFG["CLASS_NUMS"], kernel_size=1)
        )

    def forward(self, x):
        self.conv1_result = self.Conv_Block1(x)
        self.conv2_result = self.Conv_Block2(self.conv1_result)
        self.conv3_result = self.Conv_Block3(self.conv2_result)
        self.conv4_result = self.Conv_Block4(self.conv3_result)
        self.conv5_result = self.Conv_Block5(self.conv4_result)

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.Conv_Block1,
                  self.Conv_Block2,
                  self.Conv_Block3,
                  self.Conv_Block4,
                  self.Conv_Block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, torch.nn.Conv2d) and isinstance(l2, torch.nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]

class FCN8s(FCNs):
    def __init__(self, cfg):
        super(FCN8s, self).__init__(cfg=cfg)

        self.score_pool4 = torch.nn.Conv2d(512, self.n_classes, 1)
        self.score_pool3 = torch.nn.Conv2d(256, self.n_classes, 1)

    def forward(self, x):
        super(FCN8s, self).forward(x)

        score = self.classifier(self.conv5_result)
        score_pool4 = self.score_pool4(self.conv4_result)
        score_pool3 = self.score_pool4(self.conv3_result)

        score = torch.nn.functional.upsample(score,size=score_pool4.size()[2:], mode='bilinear')
        score += score_pool4

        score = torch.nn.functional.upsample(score,size=score_pool3.size()[2:], mode='bilinear')
        score += score_pool3

        output = torch.nn.functional.upsample(score, size=x.size()[2:], mode='bilinear')
        return output

class FCN16s(FCNs):
    def __init__(self, cfg):
        super(FCN16s, self).__init__(cfg)

        self.score_pool4 = torch.nn.Conv2d(512, self.n_classes, 1)

    def forward(self, x):
        super(FCN16s, self).forward(x)

        score = self.classifier(self.conv5_result)
        score_pool4 = self.score_pool4(self.conv4_result)

        score = torch.nn.functional.upsample(score,size=score_pool4.size()[2:], mode='bilinear')
        score += score_pool4
        output = torch.nn.functional.upsample(score, size=x.size()[2:], mode='bilinear')
        return output

class FCN32s(FCNs):
    def __init__(self, cfg):
        super(FCN32s, self).__init__(cfg)

    def forward(self, x):
        super(FCN32s, self).forward(x)

        score = self.classifier(self.conv5_result)

        output = torch.nn.functional.upsample(score, size=x.size()[2:], mode='bilinear')
        return output