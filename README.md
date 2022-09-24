# ShuffleNetV1

## ShuffleNetV1 描述

ShuffleNetV1是旷视科技提出的一种计算高效的CNN模型，主要目的是应用在移动端，所以模型的设计目标就是利用有限的计算资源来达到最好的模型精度。ShuffleNetV1的设计核心是引入了两种操作：pointwise group convolution和channel shuffle，这在保持精度的同时大大降低了模型的计算量。因此，ShuffleNetV1和MobileNet类似，都是通过设计更高效的网络结构来实现模型的压缩和加速。

[论文](https://arxiv.org/abs/1707.01083)：Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun. "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2018.

## 模型架构

ShuffleNetV1的核心部分被分成三个阶段，每个阶段重复堆积了若干个ShuffleNetV1的基本单元。其中每个阶段中第一个基本单元采用步长为2的pointwise group convolution使特征图的width和height各降低一半，同时channels增加一倍；之后的基本单元步长均为1，保持特征图和通道数不变。此外，ShuffleNetV1中的每个基本单元中都加入了channel shuffle操作，以此来对group convolution之后的特征图进行通道维度的重组，使信息可以在不同组之间传递。

## 训练过程

```shell
  export CUDA_VISIBLE_DEVICES=0
  python train.py --model shufflev1 --data_url ./dataset/imagenet

```

```log
epoch time: 99854.980, per step time: 79.820, avg loss: 4.093
epoch time: 99863.734, per step time: 79.827, avg loss: 4.010
epoch time: 99859.792, per step time: 79.824, avg loss: 3.869
epoch time: 99840.800, per step time: 79.809, avg loss: 3.934
epoch time: 99864.092, per step time: 79.827, avg loss: 3.442
```

## 评估过程

```shell
python validate.py --model shufflenetv1 --data_url ./dataset/imagenet --checkpoint_path=[CHECKPOINT_PATH]

```

```log
result:{'Loss': 2.0479587888106323, 'Top_1_Acc': 0.7385817307692307, 'Top_5_Acc': 0.9135817307692308}, ckpt:'/home/shufflenetv1/train_parallel0/checkpoint/shufflenetv1-250_1251.ckpt', time: 98560.63866615295
```
