# YOLOv2 介绍
## 1. Yolov2 原理简介
### 1.1 网络架构
相较于V1，V2改GooleNet为Darknet-19，能够提取更丰富的特征，从而提升检测精度。删除了全连接层，仅保留卷积层。
* **细粒度特征融合（改进点1）**：将Darknet-19中第13层卷积输出（细粒度特征）与第19层卷积输出（粗粒度特征）通过 Passthrough 层融合，保留多尺度信息。

![alt text](images/image1.png)

由于只有卷积层，可输入不同尺寸图片（320–608，步长 32），网格数 $ S=\frac{图片尺寸}{32} $



## 算法概述
- 单阶段、端到端目标检测：将检测视为密集预测任务，在单个前向传播中同时输出位置、置信度与类别。
- Anchor 机制与直接位置预测：使用 K-Means（IoU 距离）聚类得到先验框（anchors），网络预测相对 anchor 的偏移；对中心坐标采用 Sigmoid 使其落在对应栅格内，训练更稳定。
- Darknet-19 骨干网络：轻量高效的 19 层卷积骨干，配合批归一化与 Leaky ReLU，在速度与准确率间取得良好折中。
- Passthrough（reorg）特征融合：将高分辨率特征重排后与低分辨率检测特征拼接，提升小目标性能。
- 多尺度训练：每隔若干迭代随机切换输入尺寸（320–608，步长 32），提升尺度泛化能力。
- 相比 YOLOv1 的主要优化：
  - 引入 anchors 与直接坐标预测（YOLOv1 直接回归绝对坐标，稳定性较差）。
  - 全面使用 BatchNorm，显著降低过拟合并提升收敛。
  - 高分辨率分类器预训练（先将分类器提升至 448，再迁移到检测）。
  - 特征融合与更优的骨干网络（Darknet-19），速度更快、精度更高。

## 技术细节
### 预测参数与核心公式
对每个栅格上的每个 anchor，网络输出：
\[
t_x,\, t_y,\, t_w,\, t_h,\, t_o,\; \{a_k\}_{k=1}^{C}
\]
其中 \(t_x, t_y\) 为中心偏移，\(t_w, t_h\) 为尺度偏移，\(t_o\) 为对象置信度的 logit，\(\{a_k\}\) 为各类别的 logit（C 为类别数）。将它们映射为实际量的公式如下：

- 边界框中心坐标（相对所在栅格的左上角）：
\[
b_x = \sigma(t_x) + c_x,\quad
b_y = \sigma(t_y) + c_y
\]
其中 \(\sigma(\cdot)\) 为 Sigmoid，\(c_x, c_y\) 为该栅格在特征图上的整数坐标。

- 边界框尺寸（相对 anchor 尺寸的指数缩放）：
\[
b_w = p_w \cdot e^{t_w},\quad
b_h = p_h \cdot e^{t_h}
\]
其中 \(p_w, p_h\) 为该 anchor 的先验宽高。

- 对象置信度（objectness）：
\[
p_{\text{obj}} = \sigma(t_o)
\]
在推理时常将“置信度”写作
\[
\text{conf} = p_{\text{obj}} \cdot \operatorname{IoU}(b, g)
\]
其中 \(g\) 为匹配的 GT 框；训练中 Darknet 实现通常以 IoU 或 1/0 目标为依据对 \(\text{conf}\) 进行回归与抑制无目标样本。

- 类别概率（Softmax）：
\[
p(k \mid \text{obj}) = \frac{e^{a_k}}{\sum_{j=1}^{C} e^{a_j}}
\]
最终每个类别的检测分数：
\[
s_k = \text{conf} \cdot p(k \mid \text{obj})
\]
后处理采用 NMS 抑制重叠框。

## 网络架构
### Darknet-19 骨干网络
- 设计思想：交替使用 \(3\times 3\) 与 \(1\times 1\) 卷积构建瓶颈结构，所有卷积层后接 BatchNorm 与 Leaky ReLU（负斜率约 0.1），下采样通过 \(2\times 2\) 最大池化。
- 典型结构（输入 \(416\times416\)）：
  - conv \(3\times3\), 32 → maxpool
  - conv \(3\times3\), 64 → maxpool
  - conv \(3\times3\), 128 → conv \(1\times1\), 64 → conv \(3\times3\), 128 → maxpool
  - conv \(3\times3\), 256 → conv \(1\times1\), 128 → conv \(3\times3\), 256 → maxpool
  - conv \(3\times3\), 512 → conv \(1\times1\), 256 → conv \(3\times3\), 512 → conv \(1\times1\), 256 → conv \(3\times3\), 512 → maxpool
  - conv \(3\times3\), 1024 → conv \(1\times1\), 512 → conv \(3\times3\), 1024 → conv \(1\times1\), 512 → conv \(3\times3\), 1024
- 分类任务末端接全局平均池化与全连接 Softmax；检测任务去掉分类头，接检测头与特征融合。

### 检测头与输出
- 在骨干末端追加若干卷积（如 \(3\times 3\) + \(1\times 1\)），最终输出通道数为 \(A \times (5 + C)\)，其中 A 为 anchor 数，5 表示 \((t_x,t_y,t_w,t_h,t_o)\)。
- 输出特征图尺寸约为 \(13\times 13\)（当输入为 \(416\times 416\)，下采样 32 倍）。

### 特征融合（Passthrough / reorg）
- 选择一层较高分辨率的特征图（例如 \(26\times26\times512\)），使用 reorg 操作将其按 \(2\times2\) 邻域重排为 \(13\times13\times2048\)。
- 与主干末端的 \(13\times13\times1024\) 检测特征在通道维拼接，得到 \(13\times13\times(1024+2048)\)。
- 再经 \(1\times1\) 卷积降维与 \(3\times3\) 卷积细化，最后输出检测张量。该融合显著提升小目标检测能力。

## 训练细节
### 输入尺寸
- 检测训练常用 \(416\times416\)；多尺度训练每隔若干迭代随机选择 \(\{320, 352, \dots, 608\}\)。
- 分类预训练先将输入提升至 \(448\times448\) 以获得更强的高分辨率特征。

### 数据增强
- 随机尺度与平移抖动（jitter），随机裁剪，水平翻转。
- 颜色扰动：HSV 颜色空间中对色相/饱和度/亮度进行随机扰动（常见幅度在 \(\pm 0.2\sim0.4\) 区间）。
- 随机填充与保持纵横比（letterbox）以适应不同输入尺寸。

### 损失函数
设正样本集合为 \(\mathcal{P}\)，\(\operatorname{IoU}_i\) 为第 \(i\) 个预测框与其 GT 的 IoU，\(\mathbf{p}_i\) 为类别概率，\(\mathbf{y}_i\) 为 one-hot 标签，整体损失：
\[
\begin{aligned}
L &= \lambda_{\text{coord}} \sum_{i \in \mathcal{P}}
\Big[(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (w_i - \hat{w}_i)^2 + (h_i - \hat{h}_i)^2\Big] \\
&\quad + \sum_i \Big[ \mathbf{1}_i^{\text{obj}} \big(\text{conf}_i - \operatorname{IoU}_i\big)^2
 + \lambda_{\text{noobj}} \mathbf{1}_i^{\text{noobj}} \big(\text{conf}_i - 0\big)^2 \Big] \\
&\quad + \sum_{i \in \mathcal{P}} \operatorname{CE}\big(\mathbf{p}_i, \mathbf{y}_i\big)
\end{aligned}
\]
- 常用权重：\(\lambda_{\text{coord}}=5\)、\(\lambda_{\text{noobj}}=0.5\)；正负样本的对象置信度分别以 IoU/0 为目标。
- 忽略阈值（ignore-thresh）：对与任一 GT IoU 较高的负样本（如 \(>0.6\)），在无目标项中不计入损失，缓解过抑制。

### 训练超参数（参考官方 Darknet 配置）
- 优化器：SGD，动量 0.9，权重衰减 0.0005。
- 学习率：初始 \(1\times10^{-3}\)（带若干百迭代的 warmup），分段衰减（如在 40k、45k 迭代处乘以 0.1）。
- 批大小与细分：batch 64，subdivisions 8 或 16（取决于显存）。
- Anchor 选择：对训练集 GT 尺寸做 K-Means（IoU 距离）得到 A 个 priors（如 A=5）。

## 性能指标
以下数据摘自“YOLO9000: Better, Faster, Stronger”（不同实现与硬件可能略有差异）：

### Pascal VOC 2007 测试集（训练于 07+12）
- YOLOv2 416：mAP ≈ 76.8，速度 ≈ 67 FPS（Titan X）
- YOLOv2 544：mAP ≈ 78.6，速度 ≈ 40 FPS
- YOLOv2 288：mAP ≈ 69.9，速度 ≈ 90+ FPS

### COCO test-dev
- AP@[.5:.95] ≈ 21–22
- AP50 ≈ 44
- 速度：以 416 输入在高端 GPU 上通常 40–60 FPS

### 与其他检测算法的对比（代表性结果）
- Faster R-CNN（ResNet-101，VOC07）：mAP ≈ 76.3，速度 ≈ 5 FPS
- SSD512（VOC07）：mAP ≈ 76–77，速度 ≈ 19 FPS
- 综合来看，YOLOv2 以显著更高的速度提供与两阶段方法相近的精度。

## 参考
- J. Redmon, A. Farhadi. YOLO9000: Better, Faster, Stronger. CVPR 2017.
- Darknet 官方实现与配置文件。

