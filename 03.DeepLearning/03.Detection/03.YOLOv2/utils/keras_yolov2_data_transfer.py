import os

ROOT = os.path.join("/input", "VOC")
IMAGE_PATH = os.path.join(ROOT, "JPEGImages")

FILE_TRAIN = os.path.join("voctrain.txt")
FILE_TEST  = os.path.join("voctest.txt")

FILE_LIST = [[FILE_TRAIN, "keras_yolov2_train.txt"],
             [FILE_TEST, "keras_yolov2_test.txt"]]

for ele in FILE_LIST:
    filepath = ele[0]
    save_to_file = ele[1]

    with open(filepath) as f:
        lines  = f.readlines()
    # 2008_000003.jpg 2 46 11 500 333 18 62 190 83 243 14
    save_to_file = open(os.path.join(ele[1]), 'w')

    for line in lines:
        splits = line.split(" ")
        image_path = os.path.join(IMAGE_PATH, splits[0])
        num_boxes = int(splits[1])

        #images/out/giraffe.jpg, 270, 163, 181, 325, 22
        line_to_save = "%s,%d,%d,%d,%d,%d\n"

        for i in range(num_boxes):
            x = int(splits[2+5*i])
            y = int(splits[3+5*i])
            x2 = int(splits[4+5*i])
            y2 = int(splits[5+5*i])
            c = int(splits[6+5*i])
            save_to_file.writelines(line_to_save % (image_path, x, y, x2 - x, y2 - y, c))

    save_to_file.close()

