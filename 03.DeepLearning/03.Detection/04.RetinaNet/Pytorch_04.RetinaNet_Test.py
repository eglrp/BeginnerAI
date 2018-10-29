import lib.pytorch.retinanet.net as rn_net
import torch
import os
import tqdm
import lib.pytorch.retinanet.predict as rn_predict

net = rn_net.RetinaNet()
stc = torch.load("results/RetinaNet_001.pth", map_location={'cuda:0': 'cpu'})
stc = dict(stc)
aa = dict()
for key,value in stc.items():
    if "tracked" not in key:
        aa[key] = value
if torch.cuda.is_available():
    net.cuda()

net.eval()
predict = rn_predict.RetinaPredict((  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'))

path = os.path.join("../testImages")
listfile = os.listdir(path)
for file in tqdm.tqdm(listfile):
    if file.endswith("jpg"):
        filename = file.split(".")[0]
        image = predict.predict(net, 0, os.path.join(path,"%s.jpg" % filename), filename, targetPath="results/")