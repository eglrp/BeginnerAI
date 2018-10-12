import os
import torch as t

import lib.yolov3.net as yolo_net
import numpy as np

weightfile = "darknet53.conv.74"
input_wh = 416
version = "voc"
save_name = "yolo3_pre.pth"

voc_config = {
    'anchors' : [[116, 90], [156, 198], [373, 326],
                 [30, 61], [62, 45], [59, 119],
                 [10, 13], [16, 30], [33, 23]],
    'root':  os.path.join("/input/VOC2012/"),
    'num_classes': 20,
    'multiscale' : True,
    'name_path' : "./model/voc.names",
    'anchors_mask' : [[0,1,2], [3,4,5], [6,7,8]]
}

def load_weights(weightfile, yolov3, version):
    load_weights_darknet53(weightfile, yolov3)

def load_weights_darknet53(weightfile, yolov3):
    fp = open(weightfile, "rb")
    header = np.fromfile(fp, dtype = np.int32, count = 5)
    weights = np.fromfile(fp, dtype = np.float32)
    print(len(weights))
    ptr = 0
    first_conv = yolov3.conv
    bn = first_conv.bn
    conv = first_conv.conv
    # first conv copy
    ptr = copy_weights(bn, conv, ptr, weights)

    layers = [yolov3.layer1, yolov3.layer2, yolov3.layer3, yolov3.layer4, yolov3.layer5]
    for layer in layers:
        for i in range(len(layer)):
            if i == 0:
                bn = layer[i].bn
                conv = layer[i].conv
                ptr = copy_weights(bn, conv, ptr, weights)
            else:
                bn = layer[i].conv1.bn
                conv = layer[i].conv1.conv
                ptr = copy_weights(bn, conv, ptr, weights)
                bn = layer[i].conv2.bn
                conv = layer[i].conv2.conv
                ptr = copy_weights(bn, conv, ptr, weights)
    print(ptr)
    fp.close()

def copy_weights(bn, conv, ptr, weights, use_bn=True):
    if use_bn:
        num_bn_biases = bn.bias.numel()

        #Load the weights
        bn_biases = t.from_numpy(weights[ptr:ptr + num_bn_biases])
        ptr += num_bn_biases

        bn_weights = t.from_numpy(weights[ptr: ptr + num_bn_biases])
        ptr  += num_bn_biases

        bn_running_mean = t.from_numpy(weights[ptr: ptr + num_bn_biases])
        ptr  += num_bn_biases

        bn_running_var = t.from_numpy(weights[ptr: ptr + num_bn_biases])
        ptr  += num_bn_biases

        #Cast the loaded weights into dims of model weights.
        bn_biases = bn_biases.view_as(bn.bias.data)
        bn_weights = bn_weights.view_as(bn.weight.data)
        bn_running_mean = bn_running_mean.view_as(bn.running_mean)
        bn_running_var = bn_running_var.view_as(bn.running_var)

        #Copy the data to model
        bn.bias.data.copy_(bn_biases)
        bn.weight.data.copy_(bn_weights)
        bn.running_mean.copy_(bn_running_mean)
        bn.running_var.copy_(bn_running_var)
    else:
        #Number of biases
        num_biases = conv.bias.numel()

        #Load the weights
        conv_biases = t.from_numpy(weights[ptr: ptr + num_biases])
        ptr = ptr + num_biases

        #reshape the loaded weights according to the dims of the model weights
        conv_biases = conv_biases.view_as(conv.bias.data)

        #Finally copy the data
        conv.bias.data.copy_(conv_biases)

    #Let us load the weights for the Convolutional layers
    num_weights = conv.weight.numel()
    conv_weights = t.from_numpy(weights[ptr:ptr+num_weights])
    ptr = ptr + num_weights

    conv_weights = conv_weights.view_as(conv.weight.data)
    conv.weight.data.copy_(conv_weights)
    return ptr

num_blocks = [1,2,8,8,4]
yolov3 = yolo_net.Darknet(num_blocks, 416)

load_weights(weightfile, yolov3, version)
# name = "convert_yolo_" + version + ".pth"
# save_path = os.path.join("./weights", name)
t.save(yolov3.state_dict(), save_name)