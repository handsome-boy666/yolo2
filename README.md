# YOLOv2 介绍
## 1. Yolov2 原理简介
相较于 YOLOv1，YOLOv2 围绕精度提升、速度保持、多类别适配三大方向做了一系列关键改进
### 1.1 锚框

#### 改进1：锚框（Anchor Boxes）
锚框是预设在特征图每个网格上的固定尺寸边框，YOLOv2 每个网格不再直接预测框的绝对尺寸，而是预测锚框的偏移量和缩放因子。每个网格预设 k 个锚框（YOLOv2 默认为 5 个）

#### 改进2：直接位置预测（Direct location prediction）
模型输出不再是直接的框坐标，而是锚框的 **x/y 偏移量**和**w/h 缩放因子**：
*  $\sigma(t_x)$ 、 $\sigma(t_y)$ 为锚框中心坐标偏移量
*  $t_w$ 、 $t_h$ 为锚框宽度、高度缩放因子

最终框坐标通过公式解码：
*  $b_x = \sigma(t_x) + c_x$ （ $c_x$ 为网格左上角 x 坐标， $\sigma(t_x)$ 限制偏移量在 0~1 之间）
*  $b_y = \sigma(t_y) + c_y$ （ $c_y$ 为网格左上角 y 坐标）
*  $b_w = p_w \cdot e^{t_w}$ （ $p_w$ 为锚框原始宽度， $t_w$ 为模型预测的宽度缩放因子）
*  $b_h = p_h \cdot e^{t_h}$ （ $p_h$ 为锚框原始高度， $t_h$ 为模型预测的高度缩放因子）

> Q：为什么要对数变换？A：平衡大小目标损失贡献

这种设计将**直接回归框尺寸**转化为**回归锚框的相对偏移**，大幅降低了回归难度，提升了框预测的稳定性。因此，输入一张图片，模型的输出为 $S \times S \times (k \times 5 + C)$，其中 $S$ 为特征图尺寸， $k$ 为每个网格预设的锚框数， $C$ 为类别数。
#### 改进3：维度聚类（Dimension Clusters）：找锚框的方法
传统锚框尺寸依赖人工经验设计（如 Faster R-CNN），YOLOv2 提出**基于 IoU 的 K-means 聚类**自动生成适配数据集的锚框：
* 从训练集中提取所有框的宽高（归一化到特征图尺寸）
* 使用k-means聚类方法选择k个锚框，兼顾召回率和模型轻量。选k=5
* 欧式距离改为 $1-IOU$ （防止大框影响过大）
<p align="center">
  <img src="images/image1.png" alt="image2" width="300"/>
</p>
这种方法通过自动学习数据分布，无需手动设计，且能有效处理不同物体尺寸的问题。

### 1.2 网络架构
#### 改进4：使用Darknet-19（全卷积网络FCN）
相较于V1，V2改GooleNet为Darknet-19，能够提取更丰富的特征，从而提升检测精度。移除了所有全连接层，仅保留卷积层。
#### 改进5：批量归一化（Batch Normalization）
在 Darknet-19 的每一层卷积后都添加了BN层，无需 Dropout 即可有效防止过拟合（直接移除dropout）
#### 改进6：细粒度特征融合（Fine-Grained Features）
将Darknet-19中第13层卷积输出（细粒度特征）与第19层卷积输出（粗粒度特征）通过 Passthrough 层融合，保留多尺度信息。
* **passthrough层**：将26×26×512的特征图按“隔点采样”方式重塑为13×13×2048（每个2×2区域压缩为1个像素，通道数扩大4倍），再与13×13×1024的特征图拼接，得到13×13×3072的融合特征。

<p align="center">
  <img src="images/image2.png" alt="image1" width="600"/>
</p>

由于只有卷积层，可输入不同尺寸图片（320–608，步长 32），网格数 $ S=\frac{图片尺寸}{32} $

### 1.3 损失函数

YOLOv2 损失函数和YOLOv1类似,但略有不同

$$
Loss = \lambda_{coord} \cdot Loss_{coord} + Loss_{obj} + \lambda_{noobj} \cdot Loss_{noobj} + Loss_{class}
$$
*  $\sigma(t_x)$ 、 $\sigma(t_y)$ 为锚框中心坐标偏移量, $\hat{t}_x$ 、 $\hat{t}_y$ 为模型预测的锚框中心坐标偏移量(取sigmoid前)
*  $t_w$ 、 $t_h$ 为锚框宽度、高度缩放因子， $\hat{t}_w$ 、 $\hat{t}_h$ 为模型预测的锚框宽度、高度缩放因子
*  $conf_{pred}$ 为模型预测的锚框置信度
*  $P_c$ 为 GT 框所属类别概率（0/1）， $\hat{P}_c$ 为模型预测的 GT 框所属类别概率

**一、坐标回归损失**：仅正样本参与，宽高对数变换平衡大小目标（权重5.0，强化定位精度）：

$$
Loss_{coord} = \sum \mathbb{1}^{obj} \left[( \sigma {(t_x)}-\sigma(\hat{t}_x))^2 + ( \sigma {(t_y)}-\sigma(\hat{t}_y))^2 + (t_w-\hat{t}_w)^2 + (t_h-\hat{t}_h)^2\right]
$$


**二、置信度损失**：有目标置信度预测（MSE）和无目标置信度预测（MSE）：

- 有目标：

$$Loss_{obj} = \sum \mathbb{1}^{obj} (\sigma(conf_{pred}) - IOU(pred, gt))^2$$

- 无目标（权重0.5，缓解正负样本不均衡）：

$$Loss_{noobj} = \sum \mathbb{1}^{noobj} (\sigma(conf_{pred}) - 0)^2$$

**三、类别损失**：仅正样本参与，单/多标签分别用交叉熵/BCE：

$$
Loss_{class} = \sum \mathbb{1}^{obj} \sum_c \left[\hat{P}_c \log(P_c) + (1-\hat{P}_c) \log(1-P_c)\right]
$$

### 1.4 训练配置

- **参数配置**：ImageNet 1000类分类数据集上训练网络160个epoch，使用随机梯度下降，起始学习率为0.1，多项式率衰减为4，权重衰减为0.0005，动量为0.9
- **数据增强**：包括随机裁剪、旋转、色调、饱和度和曝光偏移
- 在224 × 224的图像上进行初始训练后，将网络调整到更大的尺寸448。使用上述参数进行训练，但只训练10个epoch，并以10−3的学习率开始。
- 训练网络160个epoch，初始学习率为10−3，在60和90 epoch时除以10。

#### 改进7：多尺度训练（Multi-Scale Training）
为让模型适应不同尺寸的目标，采用动态调整输入分辨率的训练策略：每隔10个batch，随机选择输入图像的分辨率（从320×320到608×608，步长为32，与网络下采样倍数一致）。

#### 改进8：高分辨率分类器（High Resolution Classifier）
先用224×224分辨率在ImageNet上预训练Darknet-19，完成分类任务收敛。再在高分辨率上训练10个epoch微调。

#### 改进9：联合训练（Joint Training）：在后续的版本验证没有用
成本高、提升有限
<p align="center">
  <img src="images/image3.png" alt="image1" width="300"/>
</p>

- **损失函数适配**
    - 当输入图像来自检测数据集时，计算完整损失（分类损失+定位损失+置信度损失）。
    - 当输入图像来自分类数据集时，仅计算分类损失，定位损失和置信度损失置零。
- **数据融合**：使用WordTree（词树），将检测数据集（如COCO，80类，含边界框标注）与分类数据集（如ImageNet，1000类，仅含类别标注）混合训练。
    - 如标签为‘动物’，反向传播只从动物这里传。如果标签为‘金毛’，反向传播从‘金毛’-‘狗’-‘动物’传播

### 1.5 总结

- YOLO2在“精度-速度”权衡上全面超越初代YOLO，同时优于Faster R-CNN和SSD等主流模型。
- 联合训练的YOLO9000在ImageNet分类任务上准确率达19.7%，在COCO检测任务上mAP达74.9%，实现“多类别识别+实时检测”的突破。

**局限性**

- 对密集小目标（如人群、密集车辆）的检测仍有漏检风险，因每个网格仅对应固定数量锚框。
- 联合训练的类别层级树构建依赖人工，对无明确层级的类别适配性差。
- 识别服装、装备能力差


## 2 本项目介绍

除了联合训练未实现，本项目与原论文方法基本一致。

## 3 本项目使用教程

本项目使用 PyTorch 实现，适配二维码数据集（单类别）和植物大战僵尸数据集（多类别）进行训练。




---

# 3 快速上手

本工程完整复现了 YOLOv2 算法，基于 PyTorch 框架。以下是针对本工程代码的详细解读与使用教程。

### 步骤 1：数据集准备

将数据集放入dataset文件夹中，本项目用到的数据集在百度网盘链接
```
dataset/YourData/
├── train/
│   ├── images/  # .jpg / .png
│   └── labels/  # .txt (YOLO格式: class x y w h)
└── test/
│   ├── images/
│   └── labels/
└── data.yaml   # 包含类别信息，如类别数、类别名称等
```

### 步骤 2：训练
修改 `config.yaml` 文件中train部分，将data_dir指向你的数据集路径，num_classes修改为数据集的类别数， num_anchors修改为数据集的锚框数（默认5），同时配置其他训练参数。

```bash
python train.py
```
运行后会自动检测或生成锚并开始训练。

### 步骤 3：推理与可视化
#### （1）预测单张图片
修改 `config.yaml` 文件中的 `predict` 部分，run_test设置为false，放置好锚框文件，设置好其他参数。
```bash
python predict.py
```
#### （2）批量测试与评估
run_test设置为true，放置好锚框文件，设置好其他参数。
```bash
python predict.py
```