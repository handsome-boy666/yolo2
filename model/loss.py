import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv2Loss(nn.Module):
    def __init__(self, anchors, num_classes, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOv2Loss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def compute_iou(self, boxes1, boxes2):
        """
        计算两组框的 IoU
        boxes1: [N, 4] (x, y, w, h)
        boxes2: [M, 4] (x, y, w, h)
        Returns: [N, M]
        """
        # 转换为左上角和右下角坐标
        b1_x1, b1_y1 = boxes1[:, 0] - boxes1[:, 2] / 2, boxes1[:, 1] - boxes1[:, 3] / 2
        b1_x2, b1_y2 = boxes1[:, 0] + boxes1[:, 2] / 2, boxes1[:, 1] + boxes1[:, 3] / 2
        
        b2_x1, b2_y1 = boxes2[:, 0] - boxes2[:, 2] / 2, boxes2[:, 1] - boxes2[:, 3] / 2
        b2_x2, b2_y2 = boxes2[:, 0] + boxes2[:, 2] / 2, boxes2[:, 1] + boxes2[:, 3] / 2

        # 计算交集区域
        inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
        inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
        inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
        inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # 计算并集区域
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]
        union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area

        return inter_area / (union_area + 1e-6) # 计算每个 GT 框与每个 Anchor 的 IoU

    def build_targets(self, pred_shape, target, anchors, device):
        """构建训练目标"""
        B, num_anchors, H, W, _ = pred_shape
        
        # 初始化 Mask 和 Target 张量
        mask = torch.zeros(B, num_anchors, H, W, device=device)
        sigmoid_tx_ty_target = torch.zeros(B, num_anchors, H, W, 2, device=device)
        tw_th_target = torch.zeros(B, num_anchors, H, W, 2, device=device)
        cls_target = torch.zeros(B, num_anchors, H, W, dtype=torch.long, device=device)
        
        # 将 anchors 缩放到网格尺度
        anchors_scaled = torch.tensor(anchors, device=device) * H  # 假设 H=W=S
        
        for b in range(B):
            # 获取当前 batch 的 GT (过滤掉 padding)
            batch_target = target[target[:, 0] == b]    # 这是当前 batch 的 GT 框
            if len(batch_target) == 0:
                continue

            # 转换 GT 坐标到网格尺度
            # batch_target: [batch_idx, class, x, y, w, h] (归一化)
            gx = batch_target[:, 2] * W
            gy = batch_target[:, 3] * H
            gw = batch_target[:, 4] * W
            gh = batch_target[:, 5] * H
            
            # 获取网格坐标
            gi = gx.long().clamp(0, W - 1)
            gj = gy.long().clamp(0, H - 1)

            # 匹配最佳 Anchor
            # 将 GT 移到原点，与 Anchors 计算 IoU
            gt_box_shifted = torch.zeros((len(batch_target), 4), device=device)
            gt_box_shifted[:, 2] = gw
            gt_box_shifted[:, 3] = gh
            
            anchor_box_shifted = torch.zeros((num_anchors, 4), device=device)
            anchor_box_shifted[:, 2] = anchors_scaled[:, 0]
            anchor_box_shifted[:, 3] = anchors_scaled[:, 1]
            
            iou = self.compute_iou(gt_box_shifted, anchor_box_shifted)
            best_anchor_idx = torch.argmax(iou, dim=1)  # 选择每个 GT 框的最佳 Anchor

            # 设置 Targets
            for i, anchor_idx in enumerate(best_anchor_idx):
                x, y = gi[i], gj[i]
                
                # 标记该位置为正样本
                mask[b, anchor_idx, y, x] = 1
                
                # 坐标偏移 Target (相对于网格左上角)
                sigmoid_tx_ty_target[b, anchor_idx, y, x, 0] = gx[i] - x.float()
                sigmoid_tx_ty_target[b, anchor_idx, y, x, 1] = gy[i] - y.float()
                
                # 尺寸缩放 Target (相对于 Anchor)，使用对数变换，因此不需要开根号计算损失
                tw_th_target[b, anchor_idx, y, x, 0] = torch.log(gw[i] / anchors_scaled[anchor_idx, 0] + 1e-16)
                tw_th_target[b, anchor_idx, y, x, 1] = torch.log(gh[i] / anchors_scaled[anchor_idx, 1] + 1e-16)
                
                # 类别 Target
                cls_target[b, anchor_idx, y, x] = batch_target[i, 1].long()

        return mask, sigmoid_tx_ty_target, tw_th_target, cls_target

    def forward(self, pred, target, S=13):
        """
        pred: [B, num_anchors * (5 + num_classes), H, W]
        target: [N, 6] -> [batch_idx, class, x, y, w, h]
        """
        B, _, H, W = pred.shape
        device = pred.device
        
        # 1. 解析预测值
        # [B, num_anchors, 5+num_classes, H, W] -> [B, num_anchors, H, W, 5+num_classes]
        pred = pred.view(B, self.num_anchors, 5 + self.num_classes, H, W)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()
        
        pred_txty = pred[..., 0:2]
        pred_twth = pred[..., 2:4]
        sigma_pred_conf = torch.sigmoid(pred[..., 4])
        pred_cls  = pred[..., 5:]

        # 2. 构建 Targets
        with torch.no_grad():
            mask, sigmoid_tx_ty_target, tw_th_target, cls_target = self.build_targets(
                pred.shape, target, self.anchors, device
            )

        # 3. 计算 Loss
        # 只计算 mask=1 (正样本) 的部分
        pos_mask = (mask == 1)
        neg_mask = (mask == 0)

        # Coordinate Loss (MSE)
        loss_x = F.mse_loss(torch.sigmoid(pred_txty[..., 0][pos_mask]), sigmoid_tx_ty_target[..., 0][pos_mask], reduction='sum')
        loss_y = F.mse_loss(torch.sigmoid(pred_txty[..., 1][pos_mask]), sigmoid_tx_ty_target[..., 1][pos_mask], reduction='sum')
        loss_w = F.mse_loss(pred_twth[..., 0][pos_mask], tw_th_target[..., 0][pos_mask], reduction='sum')   # 对数偏移
        loss_h = F.mse_loss(pred_twth[..., 1][pos_mask], tw_th_target[..., 1][pos_mask], reduction='sum')
        loss_coord = self.lambda_coord * (loss_x + loss_y + loss_w + loss_h)

        # Confidence Loss (MSE)
        # Object (target=1)
        loss_conf_obj = F.mse_loss(sigma_pred_conf[pos_mask], torch.ones_like(sigma_pred_conf[pos_mask]), reduction='sum')
        # No Object (target=0)
        loss_conf_noobj = F.mse_loss(sigma_pred_conf[neg_mask], torch.zeros_like(sigma_pred_conf[neg_mask]), reduction='sum')
        loss_conf = loss_conf_obj + self.lambda_noobj * loss_conf_noobj

        # Class Loss (CrossEntropy)
        if pos_mask.sum() > 0:
            loss_cls = F.cross_entropy(
                pred_cls[pos_mask], 
                cls_target[pos_mask], 
                reduction='sum'
            )
        else:
            loss_cls = torch.tensor(0.0, device=device)

        total_loss = loss_coord + loss_conf + loss_cls
        return total_loss / B


def yolo_v2_loss(pred, target, anchors, num_classes, S, lambda_coord=5.0, lambda_noobj=0.5):
    """函数式接口"""
    criterion = YOLOv2Loss(anchors, num_classes, lambda_coord, lambda_noobj)
    return criterion(pred, target, S)
