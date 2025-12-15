import torch
import torch.nn as nn
from model.darknet import Darknet19, Conv2dBatchLeaky

class ReorgLayer(nn.Module):    # 重组织层（对应YOLOv2的Passthrough层）：将特征图空间维度重组织，实现尺寸匹配
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride  # 重组织步长，默认2（将特征图尺寸缩小2倍，通道数扩大4倍）

    def forward(self, x):
        # 获取输入特征图尺寸：B=批次大小, C=通道数, H=高度, W=宽度
        B, C, H, W = x.data.size()
        ws = self.stride    # 重组织宽度步长（默认2）
        hs = self.stride    # 重组织高度步长（默认2）
        
        # 第一步：拆分维度并调整顺序 → 将H拆分为(H/hs, hs)，W拆分为(W/ws, ws)，并交换维度
        x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(3, 4).contiguous()
        # 第二步：重塑维度并转置 → 合并空间维度，将拆分出的stride维度移至通道方向
        x = x.view(B, C, H // hs * W // ws, hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, H // hs * W // ws, hs * ws).transpose(2, 3).contiguous()
        # 第三步：最终重塑 → 将stride维度合并到通道维度，完成空间重组织
        x = x.view(B, C * hs * ws, H // hs, W // ws)
        return x

class YOLOv2(nn.Module):
    def __init__(self, num_classes, anchors:list=None):
        super(YOLOv2, self).__init__()
        self.num_classes = num_classes  # 检测的类别数量
        self.anchors = anchors
        self.num_anchors = len(self.anchors)  # 锚框数量
        
        self.backbone = Darknet19()
        
        self.conv19 = Conv2dBatchLeaky(1024, 1024, 3, 1)
        self.conv20 = Conv2dBatchLeaky(1024, 1024, 3, 1)
        self.conv21 = Conv2dBatchLeaky(1024 + 64*4, 1024, 3, 1)
        # 输出层：生成预测结果，通道数=锚框数×(5+类别数)（5=xywh+置信度）
        self.conv22 = nn.Conv2d(1024, self.num_anchors * (5 + num_classes), 1, 1, 0)

        # pass through 层
        self.reorg = ReorgLayer(stride=2)  # 初始化重组织层，处理浅层特征尺寸
        self.conv_reorg = Conv2dBatchLeaky(512, 64, 1, 1)   # 浅层特征降维至64通道

    def forward(self, x):
        out, route1 = self.backbone(x)
        
        # 处理深层特征：强化语义信息
        out = self.conv19(out)
        out = self.conv20(out)
        
        # 处理浅层特征（用于多尺度融合）
        route1 = self.conv_reorg(route1)  # 浅层特征降维至64通道
        route1 = self.reorg(route1)   # 重组织调整尺寸，匹配深层特征
        
        # 特征融合：在通道维度拼接浅层特征和深层特征
        out = torch.cat([route1, out], 1)
        
        # 融合后特征提取 + 生成最终预测
        out = self.conv21(out)
        out = self.conv22(out)
        
        return out  # 返回YOLOv2预测结果，形状为[B, num_anchors×(5+num_classes), H/32, W/32]

    def save_weights(self, path):
        """保存模型权重到指定路径"""
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        """从指定路径加载模型权重"""
        self.load_state_dict(torch.load(path))