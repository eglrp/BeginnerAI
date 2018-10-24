import lib_keras.yolov3.predict as k_pred
import os
import PIL

CONFIG = {
    "MODEL_PATH" : "utils/keras_yolov3.h5",
    "ANCHORS" : "10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326",
    "CLASSES" : ["person","bicycle","car" ,"motorbike","aeroplane","bus","train","truck",
                 "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
                 "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
                 "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
                 "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
                 "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
                 "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
                 "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote",
                 "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock",
                 "vase","scissors","teddy bear","hair drier","toothbrush"],
    "SCORE" : 0.3,
    "IOU" : 0.45,
    "IMAGE_SIZE" : 416,
    "TEST_IMAGE" : "../testImages/"
}
yolo = k_pred.YOLO(CONFIG)

imagepath = os.path.join(CONFIG["TEST_IMAGE"])

for file in os.listdir(imagepath):
    if file.endswith("jpg"):
        filename = file.split(".")[0]
    try:
        image = PIL.Image.open(os.path.join(imagepath, file))
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image)

        r_image.save(os.path.join("outputs", file))
yolo.close_session()


