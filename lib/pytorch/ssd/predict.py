import cv2
import torch as t
import numpy as np
import lib.utils.drawutils as draw

class SSDPredict(object):
    def __init__(self, VOC_CLASS):
        self.Color = [[0, 0, 0],
                      [128, 0, 0],
                      [0, 128, 0],
                      [128, 128, 0],
                      [0, 0, 128],
                      [128, 0, 128],
                      [0, 128, 128],
                      [128, 128, 128],
                      [64, 0, 0],
                      [192, 0, 0],
                      [64, 128, 0],
                      [192, 128, 0],
                      [64, 0, 128],
                      [192, 0, 128],
                      [64, 128, 128],
                      [192, 128, 128],
                      [0, 64, 0],
                      [128, 64, 0],
                      [0, 192, 0],
                      [128, 192, 0],
                      [0, 64, 128]]
        self.VOC_CLASS = VOC_CLASS

    def predict(self, model, epoch, sourcePath, name, targetPath="outputs/"):
        image = cv2.imread(sourcePath, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        x = cv2.resize(image, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = t.from_numpy(x).permute(2, 0, 1)

        xx = t.autograd.Variable(x.unsqueeze(0))  # 扩展第0维，因为网络的输入要求是一个batch
        if t.cuda.is_available():
            xx = xx.cuda()
        y = model(xx)
        detections = y.data
        scale = t.Tensor(rgb_image.shape[1::-1]).repeat(2)

        result = self._get_resultbox(detections, scale)
        draw.draw_box_by_cv2(image, result, '%s/SSD_%s_%03d.png' % (targetPath,name, epoch))
        # for left_up, right_bottom, class_name, _, prob in result:
        #     color = self.Color[self.VOC_CLASS.index(class_name)]
        #     cv2.rectangle(image,left_up,right_bottom,color,2)
        #     label = class_name+str(round(prob,2))
        #     text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        #     p1 = (left_up[0], left_up[1]- text_size[1])
        #     cv2.rectangle(image, (int(p1[0] - 2//2), int(p1[1] - 2 - baseline)), (int(p1[0] + text_size[0]), int(p1[1] + text_size[1])), color, -1)
        #     cv2.putText(image, label, (int(p1[0]), int(p1[1] + baseline)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)
        #
        # cv2.imwrite('%s/SSD_%s_%03d.png' % (targetPath,name, epoch),image)
        return image
    def _get_resultbox(self, detections, scale):
        result_box = []
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()

                result_box.append(
                    [
                        (pt[0], pt[1]),
                        (pt[2], pt[3]),
                        self.VOC_CLASS[i - 1],
                        float(detections[0, i, j, 0].cpu().numpy())
                    ]
                )
                j +=1
        return result_box