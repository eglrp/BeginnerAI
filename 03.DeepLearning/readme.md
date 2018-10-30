深度学习
===
本节包含图片的分类、识别、分割，自编码器以及各种对抗生成网络。

# 1.基础

# 2.图片分类

# 3.图片识别
PTH和生成动画都放在[Link](https://pan.baidu.com/s/1cX05e5wlB_2TAuCBANjykQ)
## YOLOV1
![images](result/03_01_YOLOV1.png)<br/>
![images](result/03_01_YOLOV1_Loss.png)

## SSD
| | 图片一 | 图片二 |
| ------| ------ | ------ |
| Pytorch | ![images](result/03_02_SSD_01.png) | ![images](result/03_02_SSD_02.png) |
| Keras | ![images](result/03_02_SSD_Keras_01.jpg) | ![images](result/03_02_SSD_Keras_02.jpg) |

## YOLOV2
| 图片一 | 图片二 |
| ------ | ------ |
| ![images](result/03_03_YOLOV2_01.png) | ![images](result/03_03_YOLOV2_02.png) |

## RetinaNet

## YOLOV3

# 4.图片分割
## FCN8s

## FCN 16s

## FCN32s

## UNet

## FRRN

## Mask RCNN(Shape)
| 图片一 | 图片二 |
| ------ | ------ |
| ![images](result/04_04_MASKRCNN_01.png) | ![images](result/04_04_MASKRCNN_02.png) |

## Mask RCNN(Ballon)
| 图片一 | 图片二 |
| ------ | ------ |
| ![images](result/04_05_MASKRCNN_01.png) | ![images](result/04_05_MASKRCNN_02.png) |

## Mask RCNN(COCO)
| 图片一 | 图片二 |
| ------ | ------ |
| ![images](result/04_06_MASKRCNN_01.png) | ![images](result/04_06_MASKRCNN_02.png) |

# 5.自编码器

# 6.人脸识别

# 7.对抗生成网络
动画GIF图都放在[link](https://pan.baidu.com/s/1BeK1BB5OeZuAJ9k4SLq1nw)
## 7.1.GAN
![images](result/07_01_Z.gif)<br/>
![images](result/07_01_batman.gif)<br/>

## 7.2.DCGAN
### $28 \times 28$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_02_DCGAN_01_MNIST_001.png) | ![images](result/07_02_DCGAN_01_MNIST_010.png) | ![images](result/07_02_DCGAN_01_MNIST_050.png) | ![images](result/07_02_DCGAN_01_MNIST_100.png) |

### $32 \times 32$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_02_DCGAN_02_Cifar10_001.png) | ![images](result/07_02_DCGAN_02_Cifar10_010.png) | ![images](result/07_02_DCGAN_02_Cifar10_050.png) | ![images](result/07_02_DCGAN_02_Cifar10_100.png) |

### $64 \times 64$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_02_DCGAN_03_Cat_001.png) | ![images](result/07_02_DCGAN_03_Cat_010.png) | ![images](result/07_02_DCGAN_03_Cat_050.png) | ![images](result/07_02_DCGAN_03_Cat_100.png) |

### $96 \times 96$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_02_DCGAN_04_AnimateFace_001.png) | ![images](result/07_02_DCGAN_04_AnimateFace_010.png) | ![images](result/07_02_DCGAN_04_AnimateFace_050.png) | ![images](result/07_02_DCGAN_04_AnimateFace_100.png) |

### $128 \times 128$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_02_DCGAN_05_Cat_001.png) | ![images](result/07_02_DCGAN_05_Cat_010.png) | ![images](result/07_02_DCGAN_05_Cat_050.png) | ![images](result/07_02_DCGAN_05_Cat_100.png) |

## 7.3.CGAN
### $28 \times 28$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_03_CGAN_01_MNIST_001.png) | ![images](result/07_03_CGAN_01_MNIST_010.png) | ![images](result/07_03_CGAN_01_MNIST_050.png) | ![images](result/07_03_CGAN_01_MNIST_100.png) |

### $32 \times 32$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_03_CGAN_02_Cifar10_001.png) | ![images](result/07_03_CGAN_02_Cifar10_010.png) | ![images](result/07_03_CGAN_02_Cifar10_050.png) | ![images](result/07_03_CGAN_02_Cifar10_100.png) |

### $64 \times 64$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_03_CGAN_03_CFA_001.png) | ![images](result/07_03_CGAN_03_CFA_010.png) | ![images](result/07_03_CGAN_03_CFA_050.png) | ![images](result/07_03_CGAN_03_CFA_100.png) |

### $96 \times 96$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_03_CGAN_04_CFA_001.png) | ![images](result/07_03_CGAN_04_CFA_010.png) | ![images](result/07_03_CGAN_04_CFA_050.png) | ![images](result/07_03_CGAN_04_CFA_100.png) |

### $128 \times 128$

## 7.4.infoGAN
### $28 \times 28$
| 1 Epoch | 30 Epochs | 50 Epochs |
| ------- | ------- | ------- |
| ![images](result/07_04_infoGAN_01_MNIST_001.png) | ![images](result/07_04_infoGAN_01_MNIST_030.png) | ![images](result/07_04_infoGAN_01_MNIST_050.png) |

### $32 \times 32$

### $64 \times 64$

### $96 \times 96$

### $128 \times 128$

## 7.5.WGAN
### $28 \times 28$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_05_WGAN_01_MNIST_001.png) | ![images](result/07_05_WGAN_01_MNIST_010.png) | ![images](result/07_05_WGAN_01_MNIST_050.png) | ![images](result/07_05_WGAN_01_MNIST_100.png) |

### $32 \times 32$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_05_WGAN_02_Cifar10_001.png) | ![images](result/07_05_WGAN_02_Cifar10_010.png) | ![images](result/07_05_WGAN_02_Cifar10_050.png) | ![images](result/07_05_WGAN_02_Cifar10_100.png) |

### $64 \times 64$

### $96 \times 96$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_05_WGAN_04_AnimateFace_001.png) | ![images](result/07_05_WGAN_04_AnimateFace_010.png) | ![images](result/07_05_WGAN_04_AnimateFace_050.png) | ![images](result/07_05_WGAN_04_AnimateFace_100.png) |

### $128 \times 128$

## 7.6.WGANGP
### $28 \times 28$
| 1 Epoch | 30 Epochs | 50 Epochs |
| ------- | ------- | ------- |
| ![images](result/07_06_WGANGP_01_MNIST_001.png) | ![images](result/07_06_WGANGP_01_MNIST_030.png) | ![images](result/07_06_WGANGP_01_MNIST_050.png) |

### $32 \times 32$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_06_WGANGP_02_Cifar10_001.png) | ![images](result/07_06_WGANGP_02_Cifar10_010.png) | ![images](result/07_06_WGANGP_02_Cifar10_050.png) | ![images](result/07_06_WGANGP_02_Cifar10_100.png) |

### $64 \times 64$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_06_WGANGP_03_Cat_001.png) | ![images](result/07_06_WGANGP_03_Cat_010.png) | ![images](result/07_06_WGANGP_03_Cat_050.png) | ![images](result/07_06_WGANGP_03_Cat_100.png) |

### $96 \times 96$

### $128 \times 128$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_06_WGANGP_05_Cat_001.png) | ![images](result/07_06_WGANGP_05_Cat_010.png) | ![images](result/07_06_WGANGP_05_Cat_050.png) | ![images](result/07_06_WGANGP_05_Cat_100.png) |

## 7.7.LSGAN
### $28 \times 28$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_07_LSGAN_01_MNIST_001.png) | ![images](result/07_07_LSGAN_01_MNIST_010.png) | ![images](result/07_07_LSGAN_01_MNIST_050.png) | ![images](result/07_07_LSGAN_01_MNIST_100.png) |

### $32 \times 32$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_07_LSGAN_02_Cifar10_001.png) | ![images](result/07_07_LSGAN_02_Cifar10_010.png) | ![images](result/07_07_LSGAN_02_Cifar10_050.png) | ![images](result/07_07_LSGAN_02_Cifar10_100.png) |

### $64 \times 64$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_07_LSGAN_03_Cat_001.png) | ![images](result/07_07_LSGAN_03_Cat_010.png) | ![images](result/07_07_LSGAN_03_Cat_050.png) | ![images](result/07_07_LSGAN_03_Cat_100.png) |

### $96 \times 96$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_07_LSGAN_04_AnimateFace_001.png) | ![images](result/07_07_LSGAN_04_AnimateFace_010.png) | ![images](result/07_07_LSGAN_04_AnimateFace_050.png) | ![images](result/07_07_LSGAN_04_AnimateFace_100.png) |

### $128 \times 128$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_07_LSGAN_05_Cat_001.png) | ![images](result/07_07_LSGAN_05_Cat_010.png) | ![images](result/07_07_LSGAN_05_Cat_050.png) | ![images](result/07_07_LSGAN_05_Cat_100.png) |

## 7.8.BEGAN
### $28 \times 28$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_08_BEGAN_01_MNIST_001.png) | ![images](result/07_08_BEGAN_01_MNIST_010.png) | ![images](result/07_08_BEGAN_01_MNIST_050.png) | ![images](result/07_08_BEGAN_01_MNIST_100.png) |

### $32 \times 32$

### $64 \times 64$

### $96 \times 96$

### $128 \times 128$

## 7.9.CLSGAN
### $28 \times 28$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_09_CLSGAN_01_MNIST_001.png) | ![images](result/07_09_CLSGAN_01_MNIST_010.png) | ![images](result/07_09_CLSGAN_01_MNIST_050.png) | ![images](result/07_09_CLSGAN_01_MNIST_100.png) |

### $32 \times 32$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_09_CLSGAN_02_Cifar10_001.png) | ![images](result/07_09_CLSGAN_02_Cifar10_010.png) | ![images](result/07_09_CLSGAN_02_Cifar10_050.png) | ![images](result/07_09_CLSGAN_02_Cifar10_100.png) |

### $64 \times 64$

### $96 \times 96$

### $128 \times 128$

## 7.10.CBEGAN
### $28 \times 28$
| 1 Epoch | 10 Epochs | 50 Epochs | 100 Epochs |
| ------- | ------- | ------- | ------- |
| ![images](result/07_10_CBEGAN_01_MNIST_001.png) | ![images](result/07_10_CBEGAN_01_MNIST_010.png) | ![images](result/07_10_CBEGAN_01_MNIST_050.png) | ![images](result/07_10_CBEGAN_01_MNIST_100.png) |

### $32 \times 32$

### $64 \times 64$

### $96 \times 96$

### $128 \times 128$



