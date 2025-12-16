import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

class YOLOv2Loss(nn.Module):
    def __init__(self, anchors, num_classes, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOv2Loss, self).__init__()
        self.anchors = anchors  # [(w, h), ...]
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
    def compute_iou(self, box1, box2):
        """
        计算IoU
        box1: (N, 4) [x, y, w, h]
        box2: (M, 4) [x, y, w, h]
        返回: (N, M)
        """
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            (box1[:, :2] - box1[:, 2:] / 2).unsqueeze(1),  # [N, 1, 2]
            (box2[:, :2] - box2[:, 2:] / 2).unsqueeze(0)   # [1, M, 2]
        )
        rb = torch.min(
            (box1[:, :2] + box1[:, 2:] / 2).unsqueeze(1),  # [N, 1, 2]
            (box2[:, :2] + box2[:, 2:] / 2).unsqueeze(0)   # [1, M, 2]
        )

        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

        area1 = box1[:, 2] * box1[:, 3]  # [N]
        area2 = box2[:, 2] * box2[:, 3]  # [M]
        
        union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter
        
        return inter / (union + 1e-6)

    def forward(self, pred: Tensor, target: Tensor, S: int = 13) -> Tensor:
        """
        YOLOv2 Loss 计算
        
        Args:
            pred: (B, num_anchors * (5 + num_classes), H, W) -> 需要reshape
            target: (N, 6) -> [batch_idx, class, x, y, w, h] (归一化坐标)
            S: 网格尺寸 (默认13)
            
        Returns:
            loss: 标量
        """
        device = pred.device
        batch_size = pred.size(0)
        
        # 1. 解析预测值
        # pred: [B, num_anchors * (5 + num_classes), H, W]
        # reshape -> [B, num_anchors, 5 + num_classes, H, W]
        # permute -> [B, num_anchors, H, W, 5 + num_classes]
        pred = pred.view(batch_size, self.num_anchors, 5 + self.num_classes, S, S)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()
        
        # 提取各个分量
        # txty: 中心点偏移 (sigmoid)
        # twth: 宽高缩放 (exp)
        # conf: 置信度 (sigmoid)
        # cls: 类别概率 (softmax/linear) -> CrossEntropyLoss需要logits
        
        pred_txty = torch.sigmoid(pred[..., 0:2])
        pred_twth = pred[..., 2:4]  # 这里先不exp，因为target构建时对应处理
        pred_conf = torch.sigmoid(pred[..., 4])
        pred_cls = pred[..., 5:]
        
        # 2. 构建 Target
        # 初始化掩码和目标张量
        mask = torch.zeros(batch_size, self.num_anchors, S, S, requires_grad=False).to(device)
        conf_mask = torch.ones(batch_size, self.num_anchors, S, S, requires_grad=False).to(device)
        tx_ty_target = torch.zeros(batch_size, self.num_anchors, S, S, 2, requires_grad=False).to(device)
        tw_th_target = torch.zeros(batch_size, self.num_anchors, S, S, 2, requires_grad=False).to(device)
        cls_target = torch.zeros(batch_size, self.num_anchors, S, S, requires_grad=False, dtype=torch.long).to(device)
        
        # 处理每个样本的目标
        # 检查 anchor 是否为归一化值 (假设如果所有 anchor 宽高都小于 1，则为归一化值)
        is_normalized = all(a[0] < 1.0 and a[1] < 1.0 for a in self.anchors)
        
        if is_normalized:
            # 如果是归一化值，转换为网格尺度
            anchors = torch.tensor(self.anchors, device=device) * S
        else:
            # 如果已经是像素值，转换为网格尺度 (除以 stride)
            # stride = img_size / S
            # 但这里我们不知道 img_size，假设 anchors 已经是网格尺度或者归一化值
            # 这是一个潜在的风险点，通常传入的 anchors 应该是相对于 Grid 的或者归一化的
            # 鉴于当前 data/anchor.py 生成的是归一化的，我们优先处理归一化情况
            anchors = torch.tensor(self.anchors, device=device)
            
        for b in range(batch_size):
            # 获取当前batch的target
            # target: [batch_idx, class, x, y, w, h]
            current_target = target[target[:, 0] == b]
            if current_target.size(0) == 0:
                continue
                
            # 转换坐标到网格尺度
            # target x,y,w,h 是归一化的 [0,1]
            gx = current_target[:, 2] * S
            gy = current_target[:, 3] * S
            gw = current_target[:, 4] * S
            gh = current_target[:, 5] * S
            
            # 获取网格索引
            gi = gx.long()
            gj = gy.long()
            
            # 限制索引在 [0, S-1] 范围内
            gi = gi.clamp(0, S - 1)
            gj = gj.clamp(0, S - 1)
            
            # 匹配最佳 Anchor
            # 计算 GT Box 和 Anchors 的 IoU
            # GT Box 移动到 (0,0) 位置计算
            gt_box_shifted = torch.zeros((current_target.size(0), 4), device=device)
            gt_box_shifted[:, 2] = gw
            gt_box_shifted[:, 3] = gh
            
            anchor_box_shifted = torch.zeros((self.num_anchors, 4), device=device)
            anchor_box_shifted[:, 2] = anchors[:, 0]
            anchor_box_shifted[:, 3] = anchors[:, 1]
            
            iou_anchors = self.compute_iou(gt_box_shifted, anchor_box_shifted) # [N, num_anchors]
            best_anchor_idx = torch.argmax(iou_anchors, dim=1) # [N]
            
            for i in range(current_target.size(0)):
                idx = best_anchor_idx[i]
                grid_x = gi[i]
                grid_y = gj[i]
                
                # 如果该位置已经被分配了（虽然不太可能，因为通常一个网格负责一个中心），
                # 但YOLOv2允许多个anchor负责同一个网格的不同物体
                
                mask[b, idx, grid_y, grid_x] = 1
                conf_mask[b, idx, grid_y, grid_x] = 0 # 有物体的conf_mask设为0，后面统一处理noobj
                
                # 计算坐标 Target
                # tx, ty 是相对于网格左上角的偏移
                tx_ty_target[b, idx, grid_y, grid_x, 0] = gx[i] - grid_x.float()
                tx_ty_target[b, idx, grid_y, grid_x, 1] = gy[i] - grid_y.float()
                
                # tw, th 是相对于 anchor 的缩放 (log space)
                # w = anchor_w * exp(tw) => tw = log(w / anchor_w)
                # 添加 epsilon 防止 log(0)
                tw_th_target[b, idx, grid_y, grid_x, 0] = torch.log(gw[i] / anchors[idx, 0] + 1e-16)
                tw_th_target[b, idx, grid_y, grid_x, 1] = torch.log(gh[i] / anchors[idx, 1] + 1e-16)
                
                # Class Target
                cls_target[b, idx, grid_y, grid_x] = current_target[i, 1].long()
                
                # 计算当前预测框与真实框的 IoU，作为 conf 的 target (可选，YOLOv2通常直接设为1)
                # 这里为了简单，通常设为 1，或者使用 Rescore
                # 如果使用 Rescore，则 conf target = IoU(pred, gt)
                # 这里先设为 1
                # conf_target[b, idx, grid_y, grid_x] = 1 
                
        # 3. 计算 Loss
        # 坐标 Loss
        # 只有 mask 为 1 的地方才计算
        # MSE Loss
        
        # tx, ty loss
        loss_x = F.mse_loss(pred_txty[..., 0][mask == 1], tx_ty_target[..., 0][mask == 1], reduction='sum')
        loss_y = F.mse_loss(pred_txty[..., 1][mask == 1], tx_ty_target[..., 1][mask == 1], reduction='sum')
        
        # tw, th loss
        loss_w = F.mse_loss(pred_twth[..., 0][mask == 1], tw_th_target[..., 0][mask == 1], reduction='sum')
        loss_h = F.mse_loss(pred_twth[..., 1][mask == 1], tw_th_target[..., 1][mask == 1], reduction='sum')
        
        loss_coord = self.lambda_coord * (loss_x + loss_y + loss_w + loss_h)
        
        # Confidence Loss
        # Object Loss (mask == 1)
        # target 为 1 (或者 IoU)
        loss_conf_obj = F.mse_loss(pred_conf[mask == 1], torch.ones_like(pred_conf[mask == 1]), reduction='sum')
        
        # No Object Loss (mask == 0)
        # 对于没有物体的 grid，我们需要抑制 confidence
        # 但我们还需要忽略那些虽然不是最佳anchor，但IoU依然很大的预测框（参见 YOLOv3，YOLOv2 论文中其实比较简单）
        # 这里实现简单的 YOLOv2 逻辑： conf_mask 默认为 1，如果是有物体的点设为 0（上面已做）
        # 但这会导致有物体的点不计算 noobj loss，这是对的。
        # 额外的：如果预测框与真实框 IoU > 0.6 但不是最佳 anchor，应该忽略 (mask=0, conf_mask=0)
        # 为了计算效率，这里先实现基础版本
        
        loss_conf_noobj = F.mse_loss(pred_conf[mask == 0], torch.zeros_like(pred_conf[mask == 0]), reduction='sum')
        
        loss_conf = loss_conf_obj + self.lambda_noobj * loss_conf_noobj
        
        # Class Loss
        # 只有 mask == 1 的地方计算
        # 使用 CrossEntropyLoss (需要 logits)
        # pred_cls [B, num_anchors, S, S, num_classes]
        # flatten -> [N, num_classes]
        
        if mask.sum() > 0:
            loss_cls = F.cross_entropy(
                pred_cls[mask == 1], 
                cls_target[mask == 1], 
                reduction='sum'
            )
        else:
            loss_cls = torch.tensor(0.0, device=device)
            
        total_loss = loss_coord + loss_conf + loss_cls
        
        return total_loss / batch_size

def yolo_v2_loss(pred: Tensor, target: Tensor, anchors, num_classes, S=13) -> Tensor:
    """
    函数式接口，方便外部调用
    anchors: list of (w, h)
    """
    criterion = YOLOv2Loss(anchors, num_classes)
    return criterion(pred, target, S)
