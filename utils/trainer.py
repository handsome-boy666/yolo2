import os
import random
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from model.loss import yolo_v2_loss
from utils.utils import DetectionMetrics
from utils.logger import get_lr, TrainingRecorder

class Trainer:
    """
    YOLOv2 训练器，负责管理训练流程、模型保存、断点续训等。
    """
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        anchors: List[Tuple[float, float]],
        num_classes: int,
        ckpt_dir: str = 'checkpoints',
        logger: logging.Logger = None,
        recorder: Optional[TrainingRecorder] = None,
        save_interval: int = 1,
        base_img_size: int = 416,
        config: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.anchors = anchors
        self.num_classes = num_classes
        self.ckpt_dir = ckpt_dir
        self.logger = logger
        self.recorder = recorder
        self.save_interval = save_interval
        self.base_img_size = base_img_size
        self.config = config or {}
        self.epochs = self.config.get('epochs')
        
        self.start_epoch = 1
        
        # 多尺度训练参数
        self.multi_scale_sizes = [sz for sz in range(320, 608 + 32, 32)]

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载模型检查点以恢复训练。
        """
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            self.logger.info("[Resume] 未找到检查点或路径为空，将从头开始训练。")
            return

        self.logger.info(f'[Resume] 检测到已有模型: {checkpoint_path}')
        
        # 简单交互：在自动化脚本中可移除或改为参数控制
        try:
            # 假设用户总是想在提供路径时继续训练，或者添加参数控制
            # 这里为了保持原有逻辑，我们检查文件是否存在
            pass
        except Exception:
            pass

        try:
            state = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state['model_state'])
            
            try:
                self.optimizer.load_state_dict(state['optimizer_state'])
            except Exception:
                self.logger.warning('[Resume] 优化器状态加载失败，使用新优化器继续。')
            
            self.start_epoch = int(state.get('epoch', 0)) + 1
            self.logger.info(f'[Resume] 从第 {self.start_epoch} 个 epoch 开始训练。')
            
        except Exception as e:
            self.logger.error(f'[Resume] 加载检查点失败: {e}')

    def save_checkpoint(self, epoch: int) -> None:
        """保存模型检查点"""
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(self.ckpt_dir, f'epoch_{epoch:03d}.pth')
        
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'anchors': self.anchors,
            'num_classes': self.num_classes
        }, path)
        
        self.logger.info(f'[Checkpoint] 已保存到: {path}')

    def train_epoch(self, epoch: int, current_img_size: int) -> float:
        """
        训练一个 Epoch。
        """
        self.model.train()
        total_loss = 0.0
        
        S = current_img_size // 32
        
        # 实时计算 metrics (注意：这是在训练集上)
        # 从配置中获取阈值，如果不存在则使用默认值
        conf_thresh = self.config.get('conf_thresh')
        iou_thresh = self.config.get('iou_thresh')
        nms_thresh = self.config.get('nms_thresh')
        
        metrics = DetectionMetrics(iou_thresh=iou_thresh, conf_thresh=conf_thresh, nms_thresh=nms_thresh)
        
        iterator = tqdm(self.dataloader, desc=f"Epoch {epoch}", leave=False)    # 进度条
        
        for batch_idx, (images, targets) in enumerate(iterator):
            # 调整图像尺寸
            if images.shape[2] != current_img_size:
                images = F.interpolate(images, size=(current_img_size, current_img_size), mode='bilinear', align_corners=False) # 调整图像尺寸
                
            images = images.to(self.device)
            targets = targets.to(self.device)
    
            preds = self.model(images)
            
            # 获取 loss 权重参数
            lambda_coord = self.config.get('lambda_coord')
            lambda_noobj = self.config.get('lambda_noobj')
            
            loss = yolo_v2_loss(preds, targets, self.anchors, self.num_classes, S, lambda_coord=lambda_coord, lambda_noobj=lambda_noobj)
    
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # 梯度裁剪，防止梯度爆炸
            self.optimizer.step()
    
            # 监控最大置信度
            with torch.no_grad():
                # preds shape: [B, A*(5+C), H, W]
                # reshape to [B, A, 5+C, H, W]
                B, _, H, W = preds.shape
                preds_reshaped = preds.view(B, len(self.anchors), 5 + self.num_classes, H, W)
                conf_preds = torch.sigmoid(preds_reshaped[:, :, 4, :, :])
                max_conf = conf_preds.max().item()

            # 更新指标
            metrics.update(preds.detach(), targets.detach(), self.anchors, self.num_classes, (current_img_size, current_img_size))
            
            # 记录 Batch Loss
            if self.recorder is not None:
                self.recorder.record_batch(epoch, batch_idx, float(loss.item()))
            
            total_loss += loss.item()
            iterator.set_postfix(loss=f"{loss.item():.4f}", max_conf=f"{max_conf:.4f}", size=f"{current_img_size}")
            
        avg_loss = total_loss / max(1, len(self.dataloader))
        
        # 计算 Epoch 级指标
        precision, recall, map_score, miou = metrics.compute()
        self.logger.info(f'Epoch [{epoch}/{self.epochs}] - loss: {avg_loss:.4f}, precision@{iou_thresh:.2f}: {precision:.3f}, recall@{iou_thresh:.2f}: {recall:.3f}, mAP: {map_score:.3f}, mIoU: {miou:.3f}')
        
        if self.recorder is not None:
            self.recorder.record_epoch(epoch, avg_loss, precision, recall, map_score, miou, get_lr(self.optimizer))
            
        return avg_loss

    def train(self, epochs: int, save_interval: int) -> None:
        """
        开始训练循环。
        
        Args:
            epochs: 总训练轮数 (如果不指定，使用初始化时的 self.epochs)
            save_interval: 保存间隔 (如果不指定，使用初始化时的 self.save_interval)
        """
        if epochs is not None:
            self.epochs = epochs
        if save_interval is not None:
            self.save_interval = save_interval
            
        self.logger.info(f"开始训练... 总 Epochs: {self.epochs}, 保存间隔: {self.save_interval}")
        
        for epoch in range(self.start_epoch, self.epochs + 1):
            # 多尺度训练：每个 Epoch 随机选择一个尺寸
            current_img_size = random.choice(self.multi_scale_sizes)
            self.logger.info(f"[Multi-Scale] Epoch {epoch}: 调整图像尺寸为 {current_img_size}x{current_img_size}")
            
            self.train_epoch(epoch, current_img_size)
            
            if epoch % self.save_interval == 0 or epoch == self.epochs:
                self.save_checkpoint(epoch)
