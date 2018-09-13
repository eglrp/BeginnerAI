import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from math import sqrt as sqrt
from itertools import product as product
from torch.autograd import Function
import os

from lib.ssd.utils import *

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    对于每个feature map，生成预测框（中心坐标及偏移量）
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        # 300
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        # 每个网格的预测框数目 （4 or 6）
        self.num_priors = len(cfg['aspect_ratios'])
        #方差
        self.variance = cfg['variance'] or [0.1]
        # 值为[38, 19, 10, 5, 3, 1]  即feature map的尺寸大小
        self.feature_maps = cfg['feature_maps']
        #  s_k 表示先验框大小相对于图片的比例，而 s_{min} 和 s_{max} 表示比例的最小值与最大值
        # min_sizes和max_sizes用来计算s_k,s_k_prime,以便计算 长宽比为1时的两个w.h
        # 各个特征图的先验框尺度 [30, 60, 111, 162, 213, 264]
        self.min_sizes = cfg['min_sizes']
        # [60, 111, 162, 213, 264, 315]
        self.max_sizes = cfg['max_sizes']
        # 感受野大小，即相对于原图的缩小倍数
        self.steps = cfg['steps']
        # 纵横比[[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.aspect_ratios = cfg['aspect_ratios']
        # True
        self.clip = cfg['clip']
        # VOC
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        # mean 是保存预测框的列表
        mean = []
        # 遍历不同feature map的尺寸大小
        for k, f in enumerate(self.feature_maps):
            # product用于求多个可迭代对象的笛卡尔积，它跟嵌套的 for 循环等价
            # repeat用于指定重复生成序列的次数。
            # 参考：http://funhacks.net/2017/02/13/itertools/
            # 即若f为2，则i,j取值为00,01,10,11。即遍历每一个可能

            # 当k=0,f=38时，range(f)的值为（0,1，...,37）则product(range(f), repeat=2)的所有取值为（0,0）（0,1）...直到（37,0）,,,（37,37）
            # 遍历一个feature map上的每一个网格
            for i, j in product(range(f), repeat=2):
                # fk 是第 k 个 feature map 的大小
                #image_size=300  steps为每层feature maps的感受野
                f_k = self.image_size / self.steps[k]
                # 单位中心unit center x,y
                # 每一个网格的中心，设置为：(i+0.5|fk|,j+0.5|fk|)，其中，|fk| 是第 k 个 feature map 的大小，同时，i,j∈[0,|fk|)
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k


                # 总体上：先添加长宽比为1的两个w、h（比较特殊），再通过循环添加其他长宽比
                # 长宽比aspect_ratio: 1
                # 真实大小rel size: min_size
                # 先验框大小相对于图片的比例
                #计算s_k 是为了求解w、h
                s_k = self.min_sizes[k]/self.image_size
                # 由于长宽比为1，则w=s_k  h=s_k
                mean += [cx, cy, s_k, s_k]

                # 对于 aspect ratio 为 1 时，还增加了一个 default box长宽比aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                # 由于长宽比为1，则w=s_k_prime  h=s_k_prime
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 其余的长宽比
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # 将mean的list转化为tensor
        output = torch.Tensor(mean).view(-1, 4)

        # clip:True 将输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量
        # 操作为  如果元素>max，则置为max。min类似
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, cfg, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    SSD模型由去掉全连接层的vgg网络为基础组成。在之后添加了多盒转化层。
    每个多盒层分支是：
        1）conv2d 获取分类置信度
        2）conv2d进行坐标位置预测
        3）相关层去产生特定于该层特征图大小的默认的预测框bounding  boxes



    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size  输入的图像尺寸
        base: VGG16 layers for input, size of either 300 or 500   经过修改的vgg网络
        extras: extra layers that feed to multibox loc and conf layers
                提供多盒定位的格外层  和 分类置信层（vgg网络后面新增的额外层）
        head: "multibox head" consists of loc and conf conv layers
                由定位和分类卷积层组成的multibox head
                (loc_layers, conf_layers)     vgg与extras中进行分类和回归的层
    """

    def __init__(self, cfg, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = cfg
        # 新定义一个类，该类的功能：对于每个feature map，生成预测框（中心坐标及偏移量）
        self.priorbox = PriorBox(self.cfg)
        # 调用forward，返回生成的预测框结果
        # 对于所有预测的feature map，存储着生成的不同长宽比的默认框（可以理解为anchor）
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        #300
        self.size = size

        # SSD network范围
        # 经过修改的vgg网络
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        # Layer层从conv4_3学习去缩放l2正则化特征
        # 论文中conv4_3 相比较于其他的layers，有着不同的 feature scale，我们使用 ParseNet 中的 L2 normalization 技术
        # 将conv4_3 feature map 中每一个位置的 feature norm scale 到 20，并且在 back-propagation 中学习这个 scale
        self.L2Norm = L2Norm(512, 20)
        # vgg网络后面新增的额外层
        self.extras = nn.ModuleList(extras)
        # vgg与extras中进行分类和回归的层
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        # 如果网络用于测试，则加入softmax和检测
        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(self.cfg, num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        前向传播

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test测试集:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train训练集:
                list of concat outputs from:
                    1: 分类层confidence layers, Shape: [batch*num_priors,num_classes]
                    2: 回归定位层localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        # sources保存 网络生成的不同层feature map结果，以便使用这些feature map来进行分类与回归
        sources = list()
        # 保存预测层不同feature map通过回归和分类网络的输出结果
        loc = list()
        conf = list()

        # 原论文中vgg的conv4_3，relu之后加入L2 Normalization正则化，然后保存feature map
        # apply vgg up to conv4_3 relu
        # 将vgg层的feature map保存
        for k in range(23):
            x = self.vgg[k](x)
        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7，即将原fc7层更改为卷积层输出的结果，经过relu之后保存结果
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        # 将新增层的feature map保存
        for k, v in enumerate(self.extras):
            # 每经过一个conv卷积，都relu一下
            x = F.relu(v(x), inplace=True)
            # 论文中隔一个conv保存一个结果
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        # permute  将tensor的维度换位  参数为换位顺序
        #contiguous 返回一个内存连续的有相同数据的tensor

        #source保存的是每个预测层的网络输出,即feature maps
        #loc 通过使用feature map去预测回归
        #conf通过使用feature map去预测分类
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        # 在给定维度上对输入的张量序列seq 进行连接操作    dimension=1表示在列上连接
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # 测试集上的输出
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds  定位的预测
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),                # conf preds  分类的预测
                self.priors.type(type(x.data))                  # default boxes  预测框
            )
        else:
            # 训练集上的输出
            output = (
                loc.view(loc.size(0), -1, 4),    # loc preds [32,8732,4] 通过网络输出的定位的预测
                conf.view(conf.size(0), -1, self.num_classes),  #conf preds [32,8732,21]  通过网络输出的分类的预测
                self.priors   # 不同feature map根据公式生成的锚结果 [8732,4]   内容为 中心点坐标和宽高
            )
        return output

# This function is derived from torchvision VGG make_layers()
# 此方法源自torchvision VGG make_layers（）
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    '''
    vgg的结构
    cfg:  vgg的结构
     '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    i: 3   输入图像通道数
    batch_norm    为False。若为True，则网络中加入batch_norm

    返回没有全连接层的vgg网络
    '''
    #保存vgg所有层
    layers = []
    #输入图像通道数
    in_channels = i
    for v in cfg:   #M与C会导致生成的feature map大小出现变化
        if v == 'M':  #最大池化层   默认floor模式
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':  #最大池化层   ceil模式   两种不同的maxpool方式    参考https://blog.csdn.net/GZHermit/article/details/79351803
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            # 卷积
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # 论文将 Pool5 layer 的参数，从 卷积核2×2步长为2  转变成 卷积核3×3 步长为1 外加一个 pad
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # 论文中将VGG的FC6 layer、FC7 layer 转成为 卷积层conv6,conv7 并从模型的FC6、FC7 上的参数，进行采样得到这两个卷积层的 参数
    #输入通道512  输出通道为1024  卷积核为3  padding为6    dilation为卷积核中元素之间的空洞大小
    # 修改Pool5 layer参数，导致感受野大小改变。所以conv6采用 atrous 算法，即孔填充算法。
    # 孔填充算法将卷积 weights 膨胀扩大，即原来卷积核是 3x3，膨胀后，可能变成 7x7 了，这样 receptive field 变大了，而 score map 也很大，即输出变成 dense
    #这么做的好处是，输出的 score map 变大了，即是 dense 的输出了，而且 receptive field 不会变小，而且可以变大。这对做分割、检测等工作非常重要。
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    #输入通道512  输出通道为1024  卷积核为3
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    #将 修改的层也加入到vgg网络中
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    '''
    vgg网络后面新增的额外层
    :param cfg:  '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    :param i:    1024  输入通道数
    :param batch_norm:  flase
    :return:
    '''
    # 添加到VGG的额外图层用于特征缩放
    layers = []
    #1024  输入通道数
    in_channels = i
    # 控制卷积核尺寸，一维数组选前一个数还是后一个数。在每次循环时flag都改变，导致网络的卷积核尺寸为1,3,1,3交替
    # False 为1，True为3
    # SSD网络图中s1指步长为1，s2指步长为2
    # 在该代码中，S代表步长为2，无S代表默认，即步长为1，所以cfg与论文网络结构完全匹配
    flag = False
    # enumerate枚举   k为下标   v为值
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    '''

    :param vgg: 经过修改后的vgg网络（去掉全连接层，修改pool5参数并添加新层）
    :param extra_layers: vgg网络后面新增的额外层
    :param cfg: '300': [4, 6, 6, 6, 4, 4],  不同部分的feature map上一个网格预测多少框
    :param num_classes: 20分类+1背景，共21类
    :return:
    '''
    # 保存所有参与预测的网络层
    loc_layers = []
    conf_layers = []
    # 传入的修改过的vgg网络用于预测的网络是21层以及 倒数第二层
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        #4是回归的坐标参数  cfg代表该层feature map上一个网格预测多少框
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        #num_classes是类别数 cfg代表该层feature map上一个网格预测多少框
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * num_classes, kernel_size=3, padding=1)]
    # [x::y] 从下标x开始，每隔y取值
    #论文中新增层也是每隔一个层添加一个预测层
    # 将新增的额外层中的预测层也添加上   start=2：下标起始位置
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    # 数字为每层feature map的层数  M代表最大池化层（默认floor模式）    C代表最大池化层（ceil模式）  (去掉vgg16的最后的 maxpool、fc、fc、fc、softmax)
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    #
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # 不同部分的feature map上一个网格预测多少框
    '512': [],
}


def build_ssd(phase, cfg):
    '''
    新建SSD模型
    '''
    # 训练或测试
    size = cfg["IMAGE_SIZE"]
    num_classes = cfg["num_classes"]
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    #当前SSD300只支持大小300×300的数据集训练
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return

    #base_： 经过修改后的vgg网络（去掉全连接层，修改pool5参数并添加新层）
    #extras_：  vgg网络后面新增的额外层
    # head_ :    (loc_layers, conf_layers)   vgg与extras中进行分类和回归的层
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),  #vgg方法返回 经过修改后的vgg网络（去掉全连接层，修改pool5参数并添加新层）
                                     add_extras(extras[str(size)], 1024), #vgg网络后面新增的额外层
                                     mbox[str(size)],  #mbox指不同部分的feature map上一个网格预测多少框
                                     num_classes)
    # phase：'train'    size：300    num_classes： 21 类别数（20类+1背景）
    return SSD(cfg, phase, size, base_, extras_, head_, num_classes)
