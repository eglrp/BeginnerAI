import xml.etree.ElementTree as ET
import os
import tqdm

VOC_CLASS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
             'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
             'sofa', 'train', 'tvmonitor']

ROOT_PATH = os.path.join("/input", "VOC")
TRAIN_FILE = os.path.join(ROOT_PATH, "MainSet", "det_train.txt")
TARGET_FILE = os.path.join(ROOT_PATH, "MainSet", "det_test.txt")
ANNO_FILE_FOLDER = os.path.join(ROOT_PATH, "Annotations")
def parse_rec(filename):
    """
    Parse a PASCAL VOC xml file
    解析一个 PASCAL VOC xml file
    将数据集从xml解析为txt  用于生成voc2007test.txt等
    """
    tree = ET.parse(filename)
    objects = []
    # 遍历一张图中的所有物体
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        #obj_struct['pose'] = obj.find('pose').text
        #obj_struct['truncated'] = int(obj.find('truncated').text)
        #obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        # 从原图左上角开始为原点，向右为x轴，向下为y轴。左上角（xmin，ymin）和右下角(xmax,ymax)
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)

    return objects

# 新建一个名为voc2012train的txt文件，准备写入数据
set = [("voctrain.txt", TRAIN_FILE),("voctest.txt", TARGET_FILE)]

for txt_file, sourceFile in set:
    img_ids = open(sourceFile).read().strip().split()
    txt_file = open(txt_file,'w')

    # 遍历所有的xml
    for img_id in tqdm.tqdm(img_ids):
        image_path = img_id.split('.')[0] + '.jpg'
        # txt 写入图像名字   非完整路径
        txt_file.write(image_path+' ')
        results = parse_rec(os.path.join(ANNO_FILE_FOLDER, "%s.xml" % img_id))
        num_obj = len(results)
        txt_file.write(str(num_obj)+' ')
        for result in results:
            class_name = result['name']
            bbox = result['bbox']
            class_name = VOC_CLASS.index(class_name)
            txt_file.write(str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+' '+str(class_name)+' ')
        txt_file.write('\n')
    txt_file.close()