import torch as t
import torchvision as tv
import math

class YoLoV1Net(t.nn.Module):
    def __init__(self, CFG, image_size=448):
        super(YoLoV1Net, self).__init__()
        self.CFG = CFG
        self.features_list = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.image_size = image_size

        self.features = self.make_layers(batch_norm=True)
        self.classifier = t.nn.Sequential(
            t.nn.Linear(512 * 7 * 7, 4096),
            t.nn.ReLU(True),
            t.nn.Dropout(),
            t.nn.Linear(4096, self.CFG["CELL_NUMS"] * self.CFG["CELL_NUMS"] * (self.CFG["EACH_CELL_BOXES"] * 5 + len(self.CFG["CLASSES"]))),
        )
        self._initialize_weights()

        vgg = tv.models.vgg16_bn(pretrained=True)
        new_state_dict = vgg.state_dict()
        dd = self.state_dict()
        for k in new_state_dict.keys():
            if k in dd.keys() and k.startswith('features'):
                dd[k] = new_state_dict[k]
        self.load_state_dict(dd)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = t.nn.functional.sigmoid(x) #归一化到0-1
        x = x.view(-1,self.CFG["CELL_NUMS"],self.CFG["CELL_NUMS"],self.CFG["EACH_CELL_BOXES"] * 5 + len(self.CFG["CLASSES"]))
        return x

    def make_layers(self, batch_norm=False):
        layers = []
        in_channels = 3
        first_flag=True
        for v in self.features_list:
            s=1
            if (v==64 and first_flag):
                s=2
                first_flag=False
            if v == 'M':
                layers += [t.nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = t.nn.Conv2d(in_channels, v, kernel_size=3, stride=s, padding=1)
                if batch_norm:
                    layers += [conv2d, t.nn.BatchNorm2d(v), t.nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, t.nn.ReLU(inplace=True)]
                in_channels = v
        return t.nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, t.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, t.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, t.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__ == '__main__':
    CONFIG = {
        "USE_GPU" : t.cuda.is_available(),
        "EPOCHS" : 120,
        "IMAGE_SIZE" : 448,
        "IMAGE_CHANNEL" : 3,
        "ALPHA" : 0.1,
        "BATCH_SIZE" : 32,
        "DATA_PATH" : "/input/VOC2012/JPEGImages",
        "IMAGE_LIST_FILE" : "utils/voc2012train.txt",
        "CELL_NUMS" : 7,
        "CLASS_NUMS" : 20,
        "BOXES_EACH_CELL" : 2,
        "LEARNING_RATE" : 0.001,
        "L_COORD" : 5,
        "L_NOOBJ" : 0.5,
        "CELL_NUMS" : 7,
        "EACH_CELL_BOXES" : 2,
        "CLASSES" : ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                     'sofa', 'train', 'tvmonitor']
    }
    print(YoLoV1Net(CONFIG))