import math
import torch
import torch.nn as nn
import torch.nn.init as init

import lib.retinanet.net as rn_net


print('Loading pretrained ResNet50 model..')
d = torch.load('utils/resnet50.pth')

print('Loading into FPN50..')
fpn = rn_net.FPN50()
dd = fpn.state_dict()
for k in d.keys():
    if not k.startswith('fc'):  # skip fc layers
        dd[k] = d[k]

print('Saving RetinaNet..')
net = rn_net.RetinaNet()
for m in net.modules():
    if isinstance(m, nn.Conv2d):
        init.normal(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            init.constant(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

pi = 0.01
init.constant(net.cls_head[-1].bias, -math.log((1-pi)/pi))

net.fpn.load_state_dict(dd)
torch.save(net.state_dict(), 'net.pth')
print('Done!')