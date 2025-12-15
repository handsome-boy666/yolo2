import torch
import torch.nn as nn
from model.darknet import Darknet19, Conv2dBatchLeaky

class ReorgLayer(nn.Module):    # 重组织层，用于将特征图重组织为不同的大小
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.data.size()
        ws = self.stride
        hs = self.stride
        x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, H // hs * W // ws, hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, H // hs * W // ws, hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C * hs * ws, H // hs, W // ws)
        return x

class YOLOv2(nn.Module):
    def __init__(self, num_classes, anchors=None):
        super(YOLOv2, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors if anchors else []
        self.num_anchors = len(self.anchors)
        
        self.backbone = Darknet19()
        
        self.conv19 = Conv2dBatchLeaky(1024, 1024, 3, 1)
        self.conv20 = Conv2dBatchLeaky(1024, 1024, 3, 1)
        
        self.conv21 = Conv2dBatchLeaky(512, 64, 1, 1)
        self.reorg = ReorgLayer(stride=2)
        
        # Concat happens in forward
        
        self.conv22 = Conv2dBatchLeaky(1024 + 64*4, 1024, 3, 1)
        self.conv23 = nn.Conv2d(1024, self.num_anchors * (5 + num_classes), 1, 1, 0)

    def forward(self, x):
        out, route1 = self.backbone(x)
        
        out = self.conv19(out)
        out = self.conv20(out)
        
        # Handle route1
        route1 = self.conv21(route1)
        route1 = self.reorg(route1)
        
        out = torch.cat([route1, out], 1)
        
        out = self.conv22(out)
        out = self.conv23(out)
        
        return out

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
