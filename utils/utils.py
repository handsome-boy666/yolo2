import torch
import numpy as np
import cv2
from typing import List, Tuple, Optional

def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    计算两组边界框的 IoU (Intersection over Union)。
    
    Args:
        box1: (N, 4) tensor, 格式为 [x1, y1, x2, y2]
        box2: (M, 4) tensor, 格式为 [x1, y1, x2, y2]
        
    Returns:
        (N, M) tensor, 包含两两之间的 IoU 值
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-16)

def non_max_suppression(prediction: torch.Tensor, conf_thres: float = 0.5, nms_thres: float = 0.4) -> List[Optional[torch.Tensor]]:
    """
    对模型预测结果执行非极大值抑制 (NMS)。
    
    Args:
        prediction: 模型输出，形状 (B, num_anchors*H*W, 5+num_classes)
        conf_thres: 置信度阈值，低于此值的检测框将被过滤
        nms_thres: NMS IoU 阈值，高于此值的检测框将被抑制(合并)
        
    Returns:
        output: List[torch.Tensor], 长度为 B。每个元素为 (N, 7) 的 tensor，
                包含 [x1, y1, x2, y2, obj_conf, cls_conf, cls_pred]
    """
    output = [None] * len(prediction)
    for image_i, image_pred in enumerate(prediction):
        # 1. 根据物体置信度进行过滤
        mask = image_pred[:, 4] >= conf_thres
        image_pred = image_pred[mask]
        
        if not image_pred.size(0):
            continue
            
        # 2. 计算综合得分 score = obj_conf * max(class_prob)
        class_conf, class_pred = torch.max(image_pred[:, 5:], 1, keepdim=True)
        score = image_pred[:, 4] * class_conf.squeeze()
        
        # 3. 创建检测结果张量: (x, y, w, h, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        
        # 4. 将 (x, y, w, h) 转换为 (x1, y1, x2, y2)
        box_corner = detections.new(detections.shape)
        box_corner[:, 0] = detections[:, 0] - detections[:, 2] / 2
        box_corner[:, 1] = detections[:, 1] - detections[:, 3] / 2
        box_corner[:, 2] = detections[:, 0] + detections[:, 2] / 2
        box_corner[:, 3] = detections[:, 1] + detections[:, 3] / 2
        detections[:, :4] = box_corner[:, :4]
        
        # 5. 执行 torchvision 的 NMS
        nms_out_index = torch.ops.torchvision.nms(
            detections[:, :4],
            score,
            nms_thres
        )
        
        output[image_i] = detections[nms_out_index]
        
    return output

def decode_yolo_output(output: torch.Tensor, anchors: List[Tuple[float, float]], num_classes: int, img_size: Tuple[int, int]) -> torch.Tensor:
    """
    将 YOLO 原始输出解码为边界框坐标。
    
    Args:
        output: 模型原始输出 (B, A*(5+C), H, W)
        anchors: 锚框列表 [(w, h), ...]
        num_classes: 类别数量
        img_size: 输入图像尺寸 (H, W)
        
    Returns:
        output: 解码后的 tensor (B, num_boxes, 5+num_classes)，
                其中坐标为 (x, y, w, h) 绝对像素值
    """
    nB = output.size(0)
    nH = output.size(2)
    nW = output.size(3)
    stride = img_size[0] / nH
    
    # 检查 anchor 是否为归一化值 (假设如果所有 anchor 宽高都小于 1，则为归一化值)
    is_normalized = all(a[0] < 1.0 and a[1] < 1.0 for a in anchors)
    
    # 归一化锚框到特征图尺度
    if is_normalized:
        # 如果是归一化值 (0-1)，先乘以 img_size 转为像素，再除以 stride
        # 或者直接乘以特征图尺寸: a_w * nW
        scaled_anchors = [(a_w * nW, a_h * nH) for a_w, a_h in anchors]
    else:
        # 如果已经是像素值，直接除以 stride
        scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in anchors]

    num_anchors = len(anchors)
    bbox_attrs = 5 + num_classes
    
    # 重塑张量
    prediction = output.view(nB, num_anchors, bbox_attrs, nH, nW).permute(0, 1, 3, 4, 2).contiguous()
    
    # Sigmoid 激活
    x = torch.sigmoid(prediction[..., 0])
    y = torch.sigmoid(prediction[..., 1])
    w = prediction[..., 2]
    h = prediction[..., 3]
    conf = torch.sigmoid(prediction[..., 4])
    pred_cls = torch.sigmoid(prediction[..., 5:])
    
    # 生成网格坐标
    grid_x = torch.arange(nW).repeat(nH, 1).view([1, 1, nH, nW]).type_as(x)
    grid_y = torch.arange(nH).repeat(nW, 1).t().view([1, 1, nH, nW]).type_as(y)
    
    # 锚框张量
    anchor_w = torch.tensor([a[0] for a in scaled_anchors]).type_as(x).view(1, num_anchors, 1, 1)
    anchor_h = torch.tensor([a[1] for a in scaled_anchors]).type_as(x).view(1, num_anchors, 1, 1)
    
    # 计算预测框
    pred_boxes = torch.zeros_like(prediction[..., :4])
    pred_boxes[..., 0] = x + grid_x
    pred_boxes[..., 1] = y + grid_y
    pred_boxes[..., 2] = torch.exp(w) * anchor_w
    pred_boxes[..., 3] = torch.exp(h) * anchor_h
    
    # 恢复到原图尺度
    pred_boxes *= stride
    
    # 拼接结果
    output = torch.cat((pred_boxes.view(nB, -1, 4), 
                        conf.view(nB, -1, 1), 
                        pred_cls.view(nB, -1, num_classes)), -1)
    return output

def plot_boxes(img: np.ndarray, boxes: np.ndarray, class_names: Optional[List[str]] = None, color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    在图像上绘制边界框。
    
    Args:
        img: 图像数组 (H, W, 3) BGR
        boxes: 边界框数组 (N, 7) [x1, y1, x2, y2, obj_conf, cls_conf, cls_pred]
        class_names: 类别名称列表
        color: 边框颜色 (B, G, R)。如果为 None，则根据类别自动生成不同颜色。
    """
    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1 = int(box[0]), int(box[1])
        x2, y2 = int(box[2]), int(box[3])
        
        cls_id = int(box[6])
        cls_conf = box[5]
        
        if color is not None:
            c = color
        else:
            # 根据类别 ID 生成固定颜色 (使用 HSV 空间生成不同颜色)
            # Golden ratio conjugate to generate distinct colors
            h = (cls_id * 0.618033988749895) % 1
            # H: 0-179, S: 200-255, V: 200-255 (保持高饱和度和亮度)
            h_deg = int(h * 180)
            # 使用 numpy 创建 1x1 像素的 HSV 图像并转换为 BGR
            mat = np.uint8([[[h_deg, 255, 255]]])
            bgr = cv2.cvtColor(mat, cv2.COLOR_HSV2BGR)[0][0]
            c = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
            
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        
        if class_names:
            label = f"{class_names[cls_id]} {cls_conf:.2f}"
            # 获取文本大小
            (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # 绘制文字背景框
            cv2.rectangle(img, (x1, y1 - h - 5), (x1 + w, y1), c, -1)
            # 绘制文字 (使用黑色以保持对比度)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
    return img

class DetectionMetrics:
    """
    用于计算检测指标 (Precision, Recall, mAP 等)。
    """
    def __init__(self, iou_thresh, conf_thresh, nms_thresh=0.45):
        """
        初始化指标计算器。
        
        Args:
            iou_thresh: IoU 阈值，用于判断预测框与真实框是否匹配
            conf_thresh: 置信度阈值，用于过滤低置信度的预测框
            nms_thresh: NMS IoU 阈值
        """
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        # 统计信息: [(TP, conf, pred_cls, target_cls), ...]
        self.stats = [] 
        self.total_targets = 0
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor, anchors: List[Tuple[float, float]], num_classes: int, img_size: Tuple[int, int]) -> None:
        """
        更新一个批次的统计信息。
        """
        # 1. 解码预测结果
        decoded = decode_yolo_output(preds, anchors, num_classes, img_size)
        
        # 2. 执行 NMS
        batch_detections = non_max_suppression(decoded, self.conf_thresh, self.nms_thresh)
        
        # 3. 匹配预测框与真实框
        for batch_i in range(len(batch_detections)):
            detections = batch_detections[batch_i] # (N_det, 7)
            
            # 获取当前图片的 targets
            target_mask = targets[:, 0] == batch_i
            batch_targets = targets[target_mask] # (N_tgt, 6)
            
            self.total_targets += len(batch_targets)
            
            if detections is None:
                continue
                
            # 将归一化的 GT 转换为绝对像素坐标
            gt_boxes = batch_targets[:, 2:6].clone()
            gt_boxes[:, 0] *= img_size[1] # x * W
            gt_boxes[:, 1] *= img_size[0] # y * H
            gt_boxes[:, 2] *= img_size[1] # w * W
            gt_boxes[:, 3] *= img_size[0] # h * H
            
            # 将 GT 转换为 xyxy 格式
            gt_xyxy = torch.zeros_like(gt_boxes)
            gt_xyxy[:, 0] = gt_boxes[:, 0] - gt_boxes[:, 2] / 2
            gt_xyxy[:, 1] = gt_boxes[:, 1] - gt_boxes[:, 3] / 2
            gt_xyxy[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2] / 2
            gt_xyxy[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3] / 2
            
            gt_classes = batch_targets[:, 1]
            
            # 匹配逻辑
            if len(gt_xyxy) == 0:
                for det in detections:
                    self.stats.append((0, det[4], det[6], -1)) # FP
                continue
                
            # 计算 IoU 矩阵 (N_det, N_gt)
            iou_matrix = box_iou(detections[:, :4], gt_xyxy)
            
            # 严格的匹配逻辑 (mAP 计算标准)：
            # 1. 按置信度降序排序
            # 2. 贪婪匹配：高置信度框优先匹配 GT
            # 3. 每个 GT 只能被匹配一次
            
            # 按置信度降序排序
            desc_indices = torch.argsort(detections[:, 4], descending=True)
            detections = detections[desc_indices]
            iou_matrix = iou_matrix[desc_indices]
            
            # 记录 GT 是否已被匹配
            gt_matched = torch.zeros(len(gt_xyxy), dtype=torch.bool, device=detections.device)
            
            for i, det in enumerate(detections):
                # 找到该预测框与所有 GT 的最大 IoU
                max_iou, max_idx = torch.max(iou_matrix[i], dim=0)
                
                if max_iou > self.iou_thresh:
                    # 如果 IoU 达标
                    gt_cls = gt_classes[max_idx]
                    
                    if not gt_matched[max_idx]:
                        # 该 GT 尚未被匹配
                        if det[6] == gt_cls:
                            # 类别正确 -> TP
                            self.stats.append((1, det[4], det[6], gt_cls))
                            gt_matched[max_idx] = True
                        else:
                            # 类别错误 -> FP
                            self.stats.append((0, det[4], det[6], gt_cls))
                    else:
                        # 该 GT 已被更优的框匹配 -> FP (重复检测)
                        self.stats.append((0, det[4], det[6], -1))
                else:
                    # IoU 不足 -> FP
                    self.stats.append((0, det[4], det[6], -1))

    def compute(self) -> Tuple[float, float, float]:
        """计算最终指标"""
        if not self.stats:
            return 0.0, 0.0, 0.0
            
        tp = sum([s[0] for s in self.stats])
        total_pred = len(self.stats)
        
        precision = tp / max(1, total_pred)
        recall = tp / max(1, self.total_targets)
        
        # 简化版 mIoU，暂未完整实现
        miou = 0.0 
        
        return precision, recall, miou
