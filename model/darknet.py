import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dBatchLeaky(nn.Module):
    """
    带批量归一化和LeakyReLU激活的卷积层
    整合了Conv2d + BatchNorm2d + LeakyReLU的组合操作
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, leaka=0.1):
        super(Conv2dBatchLeaky, self).__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=leaka)
        )

    def forward(self, x):
        return self.block(x)

class Darknet19(nn.Module):
    """
    Darknet19骨干网络
    YOLOv2使用的特征提取网络，包含19个卷积层，以最大池化实现下采样
    输出：深层特征图 + 浅层特征图（用于后续多尺度特征融合）
    """
    def __init__(self):
        super(Darknet19, self).__init__()
        self.block1 = nn.Sequential(
            Conv2dBatchLeaky(3, 32, 3, 1),  # 卷积层1
            nn.MaxPool2d(2, 2),
            Conv2dBatchLeaky(32, 64, 3, 1),  # 卷积层2
            nn.MaxPool2d(2, 2),
            Conv2dBatchLeaky(64, 128, 3, 1),  # 卷积层3
            Conv2dBatchLeaky(128, 64, 1, 1),  # 卷积层4
            Conv2dBatchLeaky(64, 128, 3, 1),  # 卷积层5
            nn.MaxPool2d(2, 2),
            Conv2dBatchLeaky(128, 256, 3, 1),  # 卷积层6
            Conv2dBatchLeaky(256, 128, 1, 1),  # 卷积层7
            Conv2dBatchLeaky(128, 256, 3, 1),  # 卷积层8
            nn.MaxPool2d(2, 2),
            Conv2dBatchLeaky(256, 512, 3, 1),  # 卷积层9
            Conv2dBatchLeaky(512, 256, 1, 1),  # 卷积层10
            Conv2dBatchLeaky(256, 512, 3, 1),  # 卷积层11
            Conv2dBatchLeaky(512, 256, 1, 1),  # 卷积层12
            Conv2dBatchLeaky(256, 512, 3, 1),  # 卷积层13
        )
        self.block2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            Conv2dBatchLeaky(512, 1024, 1, 1),  # 卷积层14
            Conv2dBatchLeaky(1024, 512, 3, 1),  # 卷积层15
            Conv2dBatchLeaky(512, 1024, 1, 1),  # 卷积层16
            Conv2dBatchLeaky(1024, 512, 3, 1),  # 卷积层17
            Conv2dBatchLeaky(512, 1024, 1, 1),  # 卷积层18
        )
        
    def forward(self, x):
        """
        前向传播过程
        Args:
            x: 输入图像张量，形状为 [B, 3, H, W]（B=批次大小，H/W=图像尺寸）
        Returns:
            x: 深层特征图（下采样32倍），形状 [B, 1024, H/32, W/32]
            route1: 浅层特征图（下采样16倍），形状 [B, 512, H/16, W/16]（用于特征融合）
        """
        x = self.block1(x)
        route1 = x
        x = self.block2(x)
        
        # 返回深层特征和浅层特征
        return x, route1